"""Combustion and efficiency of boilers and fired heaters by the heat-loss method (bloque M18).

What it answers: stoichiometric and actual air, flue gas (wet and dry) per unit of fuel, HHV and LHV, water dew point of the
flue gas, adiabatic flame temperature, efficiency on HHV and LHV with the loss breakdown, and fuel and air for a duty.

Property data (nothing is typed here):
  * Enthalpies of every species (fuel gases and flue gas) from the NASA Glenn 9-coefficient polynomials that ship with
    Cantera (`nasa_gas.yaml`, `nasa_condensed.yaml` for liquid water). Heating values of gases are computed from those
    enthalpies of formation at 298.15 K; the flue-gas sensible heat is the enthalpy difference, never a constant cp.
  * Dry air: the composition of Cantera's `air.yaml` (O2 0.21, N2 0.78, Ar 0.01) unless the caller gives another.
  * Saturation pressure of water (relative humidity, dew point): IAPWS-95 through CoolProp.
  * Adiabatic flame with dissociation: the existing Cantera route (`packages.kinetics.cantera_engine.adiabatic_flame`).

Balance, per kg of fuel as fired, reference 298.15 K with water as vapour:
    absorbed = LHV + sensible(fuel) + sensible(moist air) - sensible(flue gas at the stack) - CO loss - setting loss
    efficiency_LHV = absorbed / LHV          efficiency_HHV = absorbed / HHV
which is the «fuel efficiency» (duty over the heating value fired), the number `utility_costs` takes as boiler_efficiency or
furnace_efficiency. `efficiency_on_total_input` divides by heating value + sensible heat of fuel and air instead.

Assumed chemistry (stated in the output): C -> CO2 (or the measured CO), H -> H2O, S -> SO2, fuel N -> N2 (no NOx), ash
inert, no unburned carbon in ash, no soot. Operating data (excess air or measured O2, stack temperature, setting loss) have no
defaults: the caller gives them."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cantera as ct
from scipy.optimize import brentq

T_REF = 298.15
R_GAS = ct.gas_constant / 1000.0           # J/(mol K)
NORMAL_M3_PER_MOL = R_GAS * 273.15 / 101325.0   # ideal gas at 0 °C and 1 atm
FLUE = ("CO2", "H2O", "N2", "O2", "Ar", "SO2", "CO", "He")
SOURCE = (f"Polinomios NASA Glenn de 9 coeficientes distribuidos con Cantera {ct.__version__} (nasa_gas.yaml; agua líquida de "
          "nasa_condensed.yaml)")

# user-friendly names -> species names of nasa_gas.yaml
ALIASES = {"methane": "CH4", "ethane": "C2H6", "propane": "C3H8", "n-butane": "C4H10,n-butane", "nC4H10": "C4H10,n-butane",
           "n-C4H10": "C4H10,n-butane", "isobutane": "C4H10,isobutane", "iC4H10": "C4H10,isobutane",
           "i-C4H10": "C4H10,isobutane", "iso-C4H10": "C4H10,isobutane", "n-pentane": "C5H12,n-pentane",
           "nC5H12": "C5H12,n-pentane", "n-C5H12": "C5H12,n-pentane", "isopentane": "C5H12,i-pentane",
           "iC5H12": "C5H12,i-pentane", "i-C5H12": "C5H12,i-pentane", "iso-C5H12": "C5H12,i-pentane", "hydrogen": "H2",
           "carbon monoxide": "CO", "carbon dioxide": "CO2", "nitrogen": "N2", "oxygen": "O2", "water": "H2O",
           "hydrogen sulfide": "H2S", "ethylene": "C2H4", "propylene": "C3H6,propylene", "C3H6": "C3H6,propylene",
           "acetylene": "C2H2,acetylene", "C2H2": "C2H2,acetylene", "AR": "Ar", "argon": "Ar", "helium": "He", "HE": "He",
           "ammonia": "NH3", "methanol": "CH3OH", "benzene": "C6H6"}
# names of the same species in GRI-Mech 3.0, for the adiabatic flame with dissociation
GRI_NAMES = {"CH4": "CH4", "C2H6": "C2H6", "C3H8": "C3H8", "H2": "H2", "CO": "CO", "CO2": "CO2", "N2": "N2", "O2": "O2",
             "H2O": "H2O", "C2H4": "C2H4", "C2H2,acetylene": "C2H2", "Ar": "AR", "NH3": "NH3", "CH3OH": "CH3OH"}
ELEMENTS = ("C", "H", "O", "N", "S")


@lru_cache(maxsize=1)
def _nasa() -> dict:
    return {s.name: s for s in ct.Species.list_from_file("nasa_gas.yaml")}


@lru_cache(maxsize=1)
def _latent_298() -> float:
    """h(H2O gas) - h(H2O liquid) at 298.15 K, J/mol (NASA: 44.004 kJ/mol)."""
    liquid = {s.name: s for s in ct.Species.list_from_file("nasa_condensed.yaml")}["H2O(L)"]
    return (_nasa()["H2O"].thermo.h(T_REF) - liquid.thermo.h(T_REF)) / 1000.0


@lru_cache(maxsize=1)
def _cantera_air() -> dict:
    air = ct.Solution("air.yaml")
    return {("Ar" if k == "AR" else k): float(x) for k, x in zip(air.species_names, air.X) if x > 0}


def _species(name: str):
    key = ALIASES.get(name, name)
    sp = _nasa().get(key)
    if sp is None:
        raise ValueError(f"especie {name!r} no está en nasa_gas.yaml; usa nombres como CH4, C2H6, C3H8, n-C4H10, i-C4H10, "
                         "n-C5H12, i-C5H12, H2, CO, H2S, C2H4, C3H6, CO2, N2, O2, H2O, Ar, He")
    return key, sp


def h_molar(name: str, T_K: float) -> float:
    """Molar enthalpy (formation + sensible) of a nasa_gas.yaml species, J/mol."""
    sp = _nasa()[name]
    if not (sp.thermo.min_temp - 2.0 <= T_K <= sp.thermo.max_temp):   # some fits start at 300 K; 298.15 K is evaluated as is
        raise ValueError(f"T = {T_K:.1f} K fuera del intervalo de los polinomios NASA de {name} "
                         f"({sp.thermo.min_temp:.0f}-{sp.thermo.max_temp:.0f} K)")
    return sp.thermo.h(T_K) / 1000.0


def _mw(name: str) -> float:
    return _nasa()[name].molecular_weight / 1000.0     # kg/mol


def _fractions(d: dict, what: str) -> dict:
    if not d or any((not isinstance(v, (int, float))) or v < 0 for v in d.values()):
        raise ValueError(f"{what}: da fracciones no negativas")
    s = sum(d.values())
    if abs(s - 1.0) > 5e-3:
        raise ValueError(f"{what}: las fracciones suman {s:.4f}; deben sumar 1 (da fracciones, no porcentajes)")
    return {k: v / s for k, v in d.items() if v > 0}


@dataclass
class Combustion:
    """Solved combustion per kg of fuel as fired. Mole numbers in mol/kg fuel."""
    fuel_kind: str
    fuel_molar_mass: float | None            # kg/mol (gas fuels)
    fuel_gas: dict                           # nasa name -> mole fraction (gas fuels)
    elements: dict                           # mol of C, H, O, N, S per kg fuel (combustible part; inerts apart)
    LHV: float                               # J/kg
    HHV: float
    heating_value_source: str
    o2_stoich: float                         # mol O2 / kg fuel
    dry_air: dict                            # mole fractions
    dry_air_mol: float                       # mol dry air / kg fuel (actual)
    moisture_per_dry_air: float              # mol H2O / mol dry air
    excess_air: float
    flue: dict                               # mol / kg fuel by species
    ash: float                               # kg / kg fuel
    fuel_sensible: float                     # J/kg fuel above 298.15 K
    air_sensible: float                      # J/kg fuel above 298.15 K (moist air)
    pressure_Pa: float
    air_T_K: float
    fuel_T_K: float | None
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # --- derived
    @property
    def stoich_dry_air_mol(self) -> float:
        return self.o2_stoich / self.dry_air["O2"]

    @property
    def dry_air_molar_mass(self) -> float:
        return sum(x * _mw(k) for k, x in self.dry_air.items())

    @property
    def air_mass(self) -> float:             # kg moist air / kg fuel
        return self.dry_air_mol * (self.dry_air_molar_mass + self.moisture_per_dry_air * _mw("H2O"))

    @property
    def flue_mol(self) -> float:
        return sum(self.flue.values())

    @property
    def flue_mass(self) -> float:            # kg flue gas / kg fuel
        return sum(n * _mw(k) for k, n in self.flue.items())

    def flue_sensible(self, T_K: float) -> float:
        """Enthalpy of the flue gas above 298.15 K with water as vapour, J per kg of fuel."""
        return sum(n * (h_molar(k, T_K) - h_molar(k, T_REF)) for k, n in self.flue.items() if n > 0)

    @property
    def co_loss(self) -> float:              # J/kg fuel: heat not released because CO did not burn to CO2
        return self.flue.get("CO", 0.0) * (h_molar("CO", T_REF) + 0.5 * h_molar("O2", T_REF) - h_molar("CO2", T_REF))

    @property
    def credits(self) -> float:              # sensible heat of fuel and air above 298.15 K, J/kg fuel
        return self.fuel_sensible + self.air_sensible

    def adiabatic_complete_T(self) -> float:
        """Temperature of the flue gas with no heat loss and no dissociation (upper bound), K."""
        target = self.LHV + self.credits - self.co_loss
        hi = min(_nasa()[k].thermo.max_temp for k, n in self.flue.items() if n > 0) - 1.0
        return brentq(lambda T: self.flue_sensible(T) - target, T_REF, hi, xtol=1e-6)


def _fuel(fuel: dict):
    if not isinstance(fuel, dict) or fuel.get("type") not in ("gas", "ultimate"):
        raise ValueError("fuel: da {'type': 'gas', 'composition': {...}, 'T_K': ...} o {'type': 'ultimate', "
                         "'mass_fractions': {'C','H','O','N','S','ash','moisture'}, 'HHV_J_kg' o 'LHV_J_kg'}")
    lat = _latent_298()
    if fuel["type"] == "gas":
        comp = {}
        for name, x in _fractions(fuel.get("composition") or {}, "fuel.composition").items():
            key, _ = _species(name)
            comp[key] = comp.get(key, 0.0) + x
        if fuel.get("T_K") is None:
            raise ValueError("fuel.T_K: da la temperatura del gas combustible en el quemador (no se supone)")
        M = sum(x * _mw(k) for k, x in comp.items())
        el = dict.fromkeys(ELEMENTS, 0.0)
        inert = {"Ar": 0.0, "He": 0.0}
        for k, x in comp.items():
            for e, a in _nasa()[k].composition.items():
                if e in el:
                    el[e] += x * a / M
                elif e in inert:
                    inert[e] += x * a / M
                else:
                    raise ValueError(f"el elemento {e} de {k} no se maneja (solo C, H, O, N, S, Ar, He)")
        lhv_molar = sum(x * h_molar(k, T_REF) for k, x in comp.items()) - M * (
            el["C"] * h_molar("CO2", T_REF) + el["H"] / 2 * h_molar("H2O", T_REF) + el["S"] * h_molar("SO2", T_REF))
        LHV = lhv_molar / M
        HHV = LHV + el["H"] / 2 * lat
        T_f = float(fuel["T_K"])
        sens = sum(x * (h_molar(k, T_f) - h_molar(k, T_REF)) for k, x in comp.items()) / M
        return dict(kind="gas", M=M, comp=comp, el=el, inert=inert, moisture=0.0, ash=0.0, LHV=LHV, HHV=HHV, T=T_f, sens=sens,
                    src=f"calculados de las entalpías de formación a 298.15 K. {SOURCE}", notes=[], warn=[])
    w = _fractions(fuel.get("mass_fractions") or {}, "fuel.mass_fractions")
    unknown = set(w) - {"C", "H", "O", "N", "S", "ash", "moisture"}
    if unknown:
        raise ValueError(f"fuel.mass_fractions: claves no reconocidas {sorted(unknown)}; usa C, H, O, N, S, ash, moisture")
    at = {e: ct.Element(e).weight / 1000.0 for e in ELEMENTS}
    el = {e: w.get(e, 0.0) / at[e] for e in ELEMENTS}
    moist = w.get("moisture", 0.0) / _mw("H2O")
    water = el["H"] / 2 + moist
    hhv, lhv = fuel.get("HHV_J_kg"), fuel.get("LHV_J_kg")
    if (hhv is None) == (lhv is None):
        raise ValueError("fuel: da HHV_J_kg o LHV_J_kg (uno solo, medido, como se quema); no se estima por correlación")
    if hhv is not None:
        HHV, LHV, given = float(hhv), float(hhv) - water * lat, "HHV"
    else:
        LHV, HHV, given = float(lhv), float(lhv) + water * lat, "LHV"
    if LHV <= 0:
        raise ValueError("el poder calorífico no es positivo")
    notes = [f"Poder calorífico: {given} dado por el usuario; el otro se obtiene con el calor latente del agua a 298.15 K "
             f"({lat/1000:.3f} kJ/mol, NASA) por el agua formada del hidrógeno y la humedad del combustible. El análisis "
             "elemental es como se quema: H y O no incluyen los de la humedad."]
    T_f, cp, sens, warn = fuel.get("T_K"), fuel.get("cp_J_kgK"), 0.0, []
    if T_f is not None and cp is not None:
        sens = float(cp) * (float(T_f) - T_REF)
    else:
        warn.append("Calor sensible del combustible NO incluido: no diste fuel.T_K y fuel.cp_J_kgK (se toma el combustible "
                    "a 298.15 K).")
    return dict(kind="ultimate", M=None, comp={}, el=el, inert={"Ar": 0.0, "He": 0.0}, moisture=moist, ash=w.get("ash", 0.0),
                LHV=LHV, HHV=HHV, T=(float(T_f) if T_f is not None else None), sens=sens,
                src=f"{given} dado por el usuario", notes=notes, warn=warn)


def _air(air: dict, pressure_Pa: float):
    if not isinstance(air, dict) or air.get("T_K") is None:
        raise ValueError("air: da {'T_K': ..., y 'humidity_ratio_kg_kg' o 'relative_humidity'} (0 si el aire es seco)")
    T = float(air["T_K"])
    dry = _fractions(air["dry_air_mole_fractions"], "air.dry_air_mole_fractions") if air.get("dry_air_mole_fractions") \
        else _cantera_air()
    dry = {ALIASES.get(k, k): v for k, v in dry.items()}
    if "O2" not in dry or set(dry) - {"O2", "N2", "Ar", "CO2"}:
        raise ValueError("air.dry_air_mole_fractions: usa O2, N2, Ar, CO2 (O2 obligatorio)")
    M_dry = sum(x * _mw(k) for k, x in dry.items())
    hr, rh = air.get("humidity_ratio_kg_kg"), air.get("relative_humidity")
    if (hr is None) == (rh is None):
        raise ValueError("air: da humidity_ratio_kg_kg (kg agua / kg aire seco) o relative_humidity (0-1), uno solo")
    if hr is not None:
        if hr < 0:
            raise ValueError("air.humidity_ratio_kg_kg no puede ser negativa")
        r = float(hr) * M_dry / _mw("H2O")
    else:
        if not 0 <= rh <= 1:
            raise ValueError("air.relative_humidity va de 0 a 1")
        import CoolProp.CoolProp as CP
        y = float(rh) * CP.PropsSI("P", "T", T, "Q", 0, "Water") / pressure_Pa
        if y >= 1:
            raise ValueError("la presión de vapor del agua a air.T_K supera la presión total")
        r = y / (1 - y)
    sens_per_mol_dry = sum(x * (h_molar(k, T) - h_molar(k, T_REF)) for k, x in dry.items()) + \
        r * (h_molar("H2O", T) - h_molar("H2O", T_REF))
    return T, dry, r, sens_per_mol_dry, air.get("dry_air_mole_fractions") is None


def solve_combustion(fuel: dict, air: dict, pressure_Pa: float, excess_air_fraction: float | None = None,
                     flue_O2: dict | None = None, flue_CO_dry_mole_fraction: float | None = None) -> Combustion:
    """Stoichiometry and enthalpy terms per kg of fuel. Exactly one of excess_air_fraction and flue_O2
    ({'mole_fraction': y, 'basis': 'dry'|'wet'}) fixes the air."""
    if pressure_Pa is None or pressure_Pa <= 0:
        raise ValueError("pressure_Pa: da la presión absoluta del sitio (no se supone 1 atm)")
    if (excess_air_fraction is None) == (flue_O2 is None):
        raise ValueError("da excess_air_fraction (0.15 = 15 %) o flue_O2 {'mole_fraction', 'basis'}, uno solo; "
                         "no se supone un exceso de aire típico")
    f = _fuel(fuel)
    T_air, dry, r, air_sens_mol, default_air = _air(air, pressure_Pa)
    el, x_o2 = f["el"], dry["O2"]
    S = el["C"] + el["H"] / 4 + el["S"] - el["O"] / 2          # mol O2 / kg fuel
    if S <= 0:
        raise ValueError("el combustible no necesita oxígeno (no es combustible)")
    y_co = float(flue_CO_dry_mole_fraction or 0.0)
    if not 0 <= y_co < 0.2:
        raise ValueError("flue_CO_dry_mole_fraction: fracción molar en base seca (100 ppm = 1e-4)")
    inert = sum(f["inert"].values())
    base = el["C"] + el["S"] + el["N"] / 2 + inert              # dry products that do not depend on the air

    def products(A: float) -> dict:
        dry_total = (base + A - S) / (1 - y_co / 2)
        n_co = y_co * dry_total
        return {"CO2": el["C"] - n_co + A * dry.get("CO2", 0.0), "CO": n_co,
                "H2O": el["H"] / 2 + f["moisture"] + A * r, "SO2": el["S"],
                "N2": el["N"] / 2 + A * dry.get("N2", 0.0), "Ar": f["inert"]["Ar"] + A * dry.get("Ar", 0.0),
                "He": f["inert"]["He"], "O2": x_o2 * A - S + n_co / 2}

    A_st = S / x_o2
    if excess_air_fraction is not None:
        e = float(excess_air_fraction)
        if e < 0 or e > 20:
            raise ValueError("excess_air_fraction debe estar entre 0 y 20 (0.15 = 15 %); con defecto de aire usa "
                             "adiabatic_flame")
        A = A_st * (1 + e)
    else:
        y, basis = flue_O2.get("mole_fraction"), flue_O2.get("basis")
        if y is None or basis not in ("dry", "wet"):
            raise ValueError("flue_O2: da {'mole_fraction': y, 'basis': 'dry' | 'wet'} (la base no se supone)")
        if not 0 <= y < x_o2:
            raise ValueError(f"flue_O2.mole_fraction debe estar entre 0 y la del aire ({x_o2})")

        def res(A_):
            p = products(A_)
            tot = sum(p.values()) - (p["H2O"] if basis == "dry" else 0.0)
            return p["O2"] / tot - y
        lo = A_st * 0.5
        while products(lo)["O2"] > 0 and lo > 1e-9:
            lo *= 0.5
        if res(A_st * 200) < 0:
            raise ValueError("el O2 medido es incompatible con este combustible y este aire")
        A = brentq(res, lo, A_st * 200, xtol=1e-12, rtol=1e-13)
        e = A / A_st - 1
    flue = products(A)
    if flue["O2"] < -1e-12 or flue["CO2"] < -1e-12:
        raise ValueError("composición imposible: con ese CO y ese aire el oxígeno o el CO2 quedan negativos")
    flue = {k: max(v, 0.0) for k, v in flue.items() if v > 1e-15}
    c = Combustion(fuel_kind=f["kind"], fuel_molar_mass=f["M"], fuel_gas=f["comp"], elements=el, LHV=f["LHV"], HHV=f["HHV"],
                   heating_value_source=f["src"], o2_stoich=S, dry_air=dry, dry_air_mol=A, moisture_per_dry_air=r,
                   excess_air=e, flue=flue, ash=f["ash"], fuel_sensible=f["sens"], air_sensible=A * air_sens_mol,
                   pressure_Pa=float(pressure_Pa), air_T_K=T_air, fuel_T_K=f["T"])
    c.assumptions += f["notes"]
    c.assumptions.append("Química supuesta: C → CO2 (o el CO medido), H → H2O, S → SO2, N del combustible → N2 (sin NOx), "
                         "ceniza inerte, sin carbono sin quemar ni hollín.")
    c.assumptions.append("Aire seco: " + ", ".join(f"{k} {v:.4f}" for k, v in dry.items()) +
                         (" (composición de air.yaml de Cantera; da air.dry_air_mole_fractions para otra)." if default_air
                          else " (dado por el usuario)."))
    c.assumptions.append("Referencia de entalpía 298.15 K con el agua como vapor; gas ideal. " + SOURCE + ".")
    c.warnings += f["warn"]
    return c


def _flame(c: Combustion) -> dict:
    out = {"T_adiabatic_complete_combustion_K": c.adiabatic_complete_T(),
           "complete_combustion_note": "Balance de entalpía propio sin disociación ni pérdidas (tope superior), con el "
                                       "combustible y el aire a sus temperaturas."}
    if c.fuel_kind != "gas":
        out["equilibrium"] = None
        out["equilibrium_note"] = "Sin llama de equilibrio: el combustible se dio por análisis elemental, no por especies."
        return out
    missing = [k for k in c.fuel_gas if k not in GRI_NAMES]
    if missing or c.flue.get("CO", 0.0) > 0 or c.dry_air.get("CO2", 0.0) > 0:
        why = (f"especies fuera de GRI-Mech 3.0: {missing}" if missing else
               "hay CO medido o CO2 en el aire; la ruta de Cantera parte de aire O2/N2/Ar y combustión completa")
        out["equilibrium"] = None
        out["equilibrium_note"] = f"Sin llama de equilibrio de Cantera: {why}."
        return out
    from packages.kinetics import cantera_engine as CE
    fuel = ", ".join(f"{GRI_NAMES[k]}:{x:.10g}" for k, x in c.fuel_gas.items())
    ox = {"O2": c.dry_air["O2"], "N2": c.dry_air.get("N2", 0.0), "AR": c.dry_air.get("Ar", 0.0),
          "H2O": c.moisture_per_dry_air}
    oxidizer = ", ".join(f"{k}:{v:.10g}" for k, v in ox.items() if v > 0)
    try:
        r = CE.adiabatic_flame(fuel, 1.0 / (1.0 + c.excess_air), T_K=c.air_T_K, P_Pa=c.pressure_Pa, oxidizer=oxidizer)
    except ValueError as err:                 # e.g. air below the 300 K lower limit of the GRI-Mech 3.0 polynomials
        out["equilibrium"] = None
        out["equilibrium_note"] = f"Sin llama de equilibrio de Cantera: {err}."
        return out
    out["equilibrium"] = {"T_adiabatic_equilibrium_K": r["T_adiabatic_equilibrium_K"],
                          "T_adiabatic_complete_combustion_K": r["T_adiabatic_complete_combustion_K"],
                          "mechanism": r["mechanism"], "engine": r["engine"], "verified": r["evidence"]["verified"],
                          "products_mole_fractions": r["products_mole_fractions"]}
    out["equilibrium_note"] = ("adiabatic_flame de Cantera (GRI-Mech 3.0, con disociación) con combustible y aire a air.T_K"
                               + ("" if c.fuel_T_K is None or abs(c.fuel_T_K - c.air_T_K) < 0.5 else
                                  f"; el combustible entra a {c.fuel_T_K:.1f} K, diferencia no incluida en esa ruta") + ".")
    return out


def _dew_point(c: Combustion) -> dict:
    import CoolProp.CoolProp as CP
    y = c.flue.get("H2O", 0.0) / c.flue_mol
    p = y * c.pressure_Pa
    T = CP.PropsSI("T", "P", p, "Q", 0, "Water") if p > 700 else None
    return {"water_mole_fraction": y, "water_partial_pressure_Pa": p, "water_dew_point_K": T,
            "note": "Punto de rocío del agua pura (IAPWS-95). El punto de rocío ácido (SO3/H2SO4) es más alto y NO se calcula."}


def combustion_efficiency(fuel: dict, air: dict, pressure_Pa: float, excess_air_fraction: float | None = None,
                          flue_O2: dict | None = None, flue_CO_dry_mole_fraction: float | None = None,
                          stack_T_K: float | None = None, setting_loss_fraction: float | None = None,
                          setting_loss_basis: str | None = None, absorbed_duty_W: float | None = None,
                          fuel_flow_kg_s: float | None = None, fuel_price_basis: str | None = None,
                          adiabatic_flame: bool = True) -> dict:
    """Combustion calculation and heat-loss efficiency; see the module docstring and tools_combustion.combustion_efficiency."""
    c = solve_combustion(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction)
    if fuel_price_basis not in (None, "HHV", "LHV"):
        raise ValueError("fuel_price_basis: 'HHV' o 'LHV'")
    M_air = c.dry_air_molar_mass
    wet, dry_mol = c.flue_mol, c.flue_mol - c.flue.get("H2O", 0.0)
    mass_in = 1.0 + c.air_mass
    mass_out = c.flue_mass + c.ash
    out = {
        "fuel": {"type": c.fuel_kind, "molar_mass_kg_kmol": c.fuel_molar_mass * 1000 if c.fuel_molar_mass else None,
                 "LHV_J_kg": c.LHV, "HHV_J_kg": c.HHV, "heating_value_source": c.heating_value_source,
                 "elements_mol_per_kg": c.elements},
        "air": {"stoichiometric_dry_air_kg_per_kg_fuel": c.stoich_dry_air_mol * M_air,
                "stoichiometric_dry_air_mol_per_kg_fuel": c.stoich_dry_air_mol,
                "excess_air_fraction": c.excess_air, "equivalence_ratio": 1.0 / (1.0 + c.excess_air),
                "dry_air_kg_per_kg_fuel": c.dry_air_mol * M_air, "moist_air_kg_per_kg_fuel": c.air_mass,
                "humidity_ratio_kg_kg": c.moisture_per_dry_air * _mw("H2O") / M_air,
                "excess_air_from": "dato del usuario" if excess_air_fraction is not None else
                                   f"O2 medido {flue_O2['mole_fraction']} en base {flue_O2['basis']} (balance exacto, no la "
                                   "fórmula O2/(21 - O2))"},
        "flue_gas": {"kg_per_kg_fuel": c.flue_mass, "mol_per_kg_fuel": wet, "molar_mass_kg_kmol": 1000 * c.flue_mass / wet,
                     "wet_mole_fractions": {k: n / wet for k, n in c.flue.items()},
                     "dry_mole_fractions": {k: n / dry_mol for k, n in c.flue.items() if k != "H2O"},
                     "ash_kg_per_kg_fuel": c.ash, **_dew_point(c)},
    }
    if c.fuel_kind == "gas":
        M = c.fuel_molar_mass
        out["fuel"].update({"LHV_J_mol": c.LHV * M, "HHV_J_mol": c.HHV * M, "LHV_J_Nm3": c.LHV * M / NORMAL_M3_PER_MOL,
                            "HHV_J_Nm3": c.HHV * M / NORMAL_M3_PER_MOL,
                            "normal_m3_definition": "0 °C y 101 325 Pa, gas ideal",
                            "composition": dict(c.fuel_gas)})
        out["air"]["stoichiometric_dry_air_mol_per_mol_fuel"] = c.stoich_dry_air_mol * M
        out["flue_gas"]["mol_per_mol_fuel"] = wet * M
    if adiabatic_flame:
        out["adiabatic_flame"] = _flame(c)
    evidence = {"mass_balance_error": abs(mass_in - mass_out) / mass_in}
    reasons = [] if evidence["mass_balance_error"] < 1e-9 else ["el balance de masa no cierra"]
    warnings = list(c.warnings)
    if stack_T_K is not None:
        if setting_loss_fraction is None or setting_loss_basis not in ("HHV", "LHV"):
            raise ValueError("con stack_T_K da setting_loss_fraction (pérdida por paredes/radiación de la envolvente como "
                             "fracción del poder calorífico; 0 si de verdad es nula) y setting_loss_basis 'HHV' o 'LHV'; no "
                             "se supone un valor típico")
        if not 0 <= setting_loss_fraction < 0.5:
            raise ValueError("setting_loss_fraction debe estar entre 0 y 0.5")
        Ts = float(stack_T_K)
        lat = _latent_298()
        water_fuel = c.flue.get("H2O", 0.0) - c.dry_air_mol * c.moisture_per_dry_air   # formed + fuel moisture
        dh = lambda k: h_molar(k, Ts) - h_molar(k, T_REF)   # noqa: E731
        dry_gas = sum(n * dh(k) for k, n in c.flue.items() if k != "H2O")
        w_formed = c.elements["H"] / 2 * dh("H2O")
        w_fuel_moist = (water_fuel - c.elements["H"] / 2) * dh("H2O")
        w_air = c.dry_air_mol * c.moisture_per_dry_air * dh("H2O")
        setting = setting_loss_fraction * (c.HHV if setting_loss_basis == "HHV" else c.LHV)
        absorbed = c.LHV + c.credits - (dry_gas + w_formed + w_fuel_moist + w_air) - c.co_loss - setting
        if absorbed <= 0:
            raise ValueError("las pérdidas superan al calor liberado: revisa la temperatura de chimenea y el exceso de aire")
        terms = {"dry_gas": dry_gas, "water_from_hydrogen_sensible": w_formed, "fuel_moisture_sensible": w_fuel_moist,
                 "air_moisture_sensible": w_air, "unburned_CO": c.co_loss, "setting": setting}
        eff = {}
        for basis, hv in (("LHV", c.LHV), ("HHV", c.HHV)):
            losses = {k: v / hv for k, v in terms.items()}
            if basis == "HHV":
                losses["water_latent"] = water_fuel * lat / hv
            cr = {"fuel_sensible": c.fuel_sensible / hv, "air_sensible": c.air_sensible / hv}
            eff[basis] = {"efficiency": absorbed / hv, "losses_fraction_of_heating_value": losses,
                          "credits_fraction_of_heating_value": cr,
                          "check_one_minus_losses_plus_credits": 1 - sum(losses.values()) + sum(cr.values()),
                          "efficiency_on_total_input": absorbed / (hv + c.credits)}
        out["efficiency"] = {"stack_T_K": Ts, "absorbed_J_per_kg_fuel": absorbed, "LHV": eff["LHV"], "HHV": eff["HHV"],
                             "definition": "efficiency = calor absorbido ÷ poder calorífico quemado (base indicada). "
                                           "efficiency_on_total_input divide entre poder calorífico + calor sensible de "
                                           "combustible y aire sobre 298.15 K.",
                             "stack_loss_only_efficiency_HHV": 1 - (sum(terms.values()) - setting - c.credits
                                                                    + water_fuel * lat) / c.HHV}
        dew = out["flue_gas"]["water_dew_point_K"]
        if dew is not None and Ts < dew:
            warnings.append(f"La chimenea ({Ts:.1f} K) está bajo el punto de rocío del agua ({dew:.1f} K): hay condensación "
                            "y este balance (agua como vapor) ya no vale; no se modelan calderas de condensación.")
        if c.fuel_kind == "gas":                      # independent closure with absolute enthalpies
            M = c.fuel_molar_mass
            h_in = sum(x * h_molar(k, c.fuel_T_K) for k, x in c.fuel_gas.items()) / M + c.dry_air_mol * (
                sum(x * h_molar(k, c.air_T_K) for k, x in c.dry_air.items())
                + c.moisture_per_dry_air * h_molar("H2O", c.air_T_K))
            h_out = sum(n * h_molar(k, Ts) for k, n in c.flue.items())
            evidence["energy_balance_error"] = abs((h_in - h_out) - (absorbed + setting)) / c.LHV
            if evidence["energy_balance_error"] > 1e-9:
                reasons.append("el balance de energía con entalpías absolutas no coincide con el de pérdidas")
        for basis in ("LHV", "HHV"):
            if abs(eff[basis]["check_one_minus_losses_plus_credits"] - eff[basis]["efficiency"]) > 1e-12:
                reasons.append("la suma de pérdidas no reproduce la eficiencia")
        out["for_utility_costs"] = {
            "boiler_or_furnace_efficiency": {"HHV": eff["HHV"]["efficiency"], "LHV": eff["LHV"]["efficiency"]},
            "rule": "Pasa a utility_costs (boiler_efficiency o furnace_efficiency) la eficiencia de la MISMA base que el precio "
                    "del combustible (prices.fuel_basis). Es un dato que tú pasas: utility_costs no llama a esta herramienta."}
        if fuel_price_basis:
            other = "LHV" if fuel_price_basis == "HHV" else "HHV"
            good, bad = eff[fuel_price_basis]["efficiency"], eff[other]["efficiency"]
            out["for_utility_costs"].update({
                "fuel_price_basis": fuel_price_basis, "efficiency_to_pass": good,
                "basis_mismatch_warning": f"Si pasaras la eficiencia en base {other} ({bad:.4f}) con un precio en base "
                                          f"{fuel_price_basis}, el costo del calor saldría {100 * (good / bad - 1):+.1f} % "
                                          "respecto al correcto."})
        if absorbed_duty_W is not None or fuel_flow_kg_s is not None:
            if absorbed_duty_W is not None and fuel_flow_kg_s is not None:
                raise ValueError("da absorbed_duty_W o fuel_flow_kg_s, no los dos")
            m = float(fuel_flow_kg_s) if fuel_flow_kg_s is not None else float(absorbed_duty_W) / absorbed
            if m <= 0:
                raise ValueError("el flujo o la carga deben ser positivos")
            out["firing"] = {"absorbed_duty_W": m * absorbed, "fuel_kg_s": m, "heat_release_LHV_W": m * c.LHV,
                             "heat_release_HHV_W": m * c.HHV, "dry_air_kg_s": m * c.dry_air_mol * M_air,
                             "moist_air_kg_s": m * c.air_mass, "flue_gas_kg_s": m * c.flue_mass,
                             "flue_gas_mol_s": m * wet, "setting_loss_W": m * setting}
            if c.fuel_kind == "gas":
                out["firing"]["fuel_mol_s"] = m / c.fuel_molar_mass
                out["firing"]["fuel_Nm3_s"] = m / c.fuel_molar_mass * NORMAL_M3_PER_MOL
    elif absorbed_duty_W is not None or setting_loss_fraction is not None:
        raise ValueError("para eficiencia o carga da stack_T_K (temperatura de chimenea medida o de diseño)")
    elif fuel_flow_kg_s is not None:
        m = float(fuel_flow_kg_s)
        out["firing"] = {"fuel_kg_s": m, "heat_release_LHV_W": m * c.LHV, "heat_release_HHV_W": m * c.HHV,
                         "dry_air_kg_s": m * c.dry_air_mol * M_air, "moist_air_kg_s": m * c.air_mass,
                         "flue_gas_kg_s": m * c.flue_mass, "flue_gas_mol_s": m * wet}
    evidence["verified"] = not reasons
    evidence["reasons"] = reasons
    out["evidence"] = evidence
    out["warnings"] = warnings
    out["assumptions"] = list(c.assumptions)
    out["not_included"] = ["carbono sin quemar en ceniza, hollín y NOx", "calor sensible de la ceniza",
                           "vapor de atomización y sopladores de hollín", "punto de rocío ácido",
                           "calderas de condensación (chimenea bajo el punto de rocío)",
                           "purga de caldera (va en utility_costs como blowdown_fraction)"]
    out["method"] = {"name": "método de pérdidas (balance de entalpía por kg de combustible)",
                     "module": "packages/energy/combustion.py", "property_data": SOURCE}
    return out
