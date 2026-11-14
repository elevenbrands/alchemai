"""Natural gas whose composition the plant does not know (bloque M20).

A plant almost never has the composition of its gas, but its supplier bills energy and volume, so the heating value per
cubic metre is a measured datum. Two routes, never silent (the caller must declare `type: 'natural_gas_unknown'`):

1. `fit`: an equivalent methane + ethane mixture (methane + nitrogen when the heating value is below methane's) whose
   molar heating value equals the measured one exactly. One measured number fixes the one free parameter.
2. `reference`: with no heating value at all, a named published composition (the reference natural gases shipped with
   CoolProp as predefined mixtures, read at run time, not retyped).

How good route 1 is gets measured, not asserted: `surrogate_error` compares, for each reference gas, the full composition
against the fitted two-component surrogate through the same combustion engine.
"""
from __future__ import annotations

from functools import lru_cache

from packages.energy import combustion as CB

FUEL_UNKNOWN_FLAG = "COMPOSICIÓN DEL GAS SUPUESTA, NO ANALIZADA"
UNKNOWN_TYPE = "natural_gas_unknown"
UNKNOWN_KEYS = ["type", "T_K", "heating_value", "reference_gas"]
HEATING_VALUE_KEYS = ["value_J_m3", "basis", "volume_reference"]
# CoolProp predefined mixtures -> short ids. Component names of CoolProp -> names the combustion engine accepts.
REFERENCE_GASES = {"GULF-COAST": "GulfCoast.mix", "AMARILLO": "Amarillo.mix", "EKOFISK": "Ekofisk.mix",
                   "HIGH-N2": "HighN2.mix", "HIGH-CO2": "HighCO2.mix", "NATURAL-GAS-SAMPLE": "NaturalGasSample.mix"}
DEFAULT_REFERENCE_GAS = "GULF-COAST"       # owner-delegated choice: see MOCKS GAS-SIN-COMPOSICION-M20
CP_TO_NASA = {"Methane": "CH4", "Nitrogen": "N2", "CarbonDioxide": "CO2", "Ethane": "C2H6", "n-Propane": "C3H8",
              "IsoButane": "i-C4H10", "n-Butane": "n-C4H10", "Isopentane": "i-C5H12", "n-Pentane": "n-C5H12",
              "n-Hexane": "n-C5H12"}     # nasa_gas.yaml has no n-hexane: lumped into n-pentane (< 0.07 % mol in every gas)
# plausibility limits of the fit (they warn, they do not refuse): the richest and leanest reference gases sit inside them
FIT_WARN_ETHANE, FIT_WARN_NITROGEN = 0.15, 0.10
FIT_REFUSE_ETHANE, FIT_REFUSE_NITROGEN = 0.50, 0.30


@lru_cache(maxsize=None)
def reference_gas(name: str) -> dict:
    """Mole fractions of one reference natural gas, read from the CoolProp predefined mixture of that name."""
    import CoolProp
    import CoolProp.CoolProp as CP

    if name not in REFERENCE_GASES:
        raise ValueError(f"fuel.reference_gas = {name!r}: no existe. Disponibles: {sorted(REFERENCE_GASES)}.")
    state = CP.AbstractState("HEOS", REFERENCE_GASES[name])
    comp: dict[str, float] = {}
    lumped = []
    for fluid, z in zip(state.fluid_names(), state.get_mole_fractions()):
        if fluid not in CP_TO_NASA:
            raise ValueError(f"el componente {fluid} de {REFERENCE_GASES[name]} no tiene equivalente en el motor de combustión")
        if fluid == "n-Hexane":
            lumped.append(f"n-hexano ({100 * z:.3f} % mol) contado como n-pentano")
        comp[CP_TO_NASA[fluid]] = comp.get(CP_TO_NASA[fluid], 0.0) + float(z)
    total = sum(comp.values())
    return {"id": name, "composition": {k: v / total for k, v in comp.items()},
            "source": f"mezcla predefinida «{REFERENCE_GASES[name]}» de CoolProp {CoolProp.__version__} (gases naturales de "
                      "referencia de NIST; composición leída de la librería, no transcrita)",
            "notes": lumped}


@lru_cache(maxsize=None)
def _pure_heating_values(species: str) -> dict:
    f = CB._fuel({"type": "gas", "composition": {species: 1.0}, "T_K": CB.T_REF})
    return {"HHV": f["HHV"] * f["M"], "LHV": f["LHV"] * f["M"]}


def molar_heating_value(composition: dict, basis: str) -> float:
    """J/mol of a gas composition on the given basis, from the same enthalpies of formation the combustion engine uses."""
    f = CB._fuel({"type": "gas", "composition": composition, "T_K": CB.T_REF})
    return (f["HHV"] if basis == "HHV" else f["LHV"]) * f["M"]


def fit_to_heating_value(heating_value_J_mol: float, basis: str) -> dict:
    """Two-component natural gas whose molar heating value equals the measured one: CH4 + C2H6, or CH4 + N2 if leaner."""
    if basis not in ("HHV", "LHV"):
        raise ValueError("basis del poder calorífico: 'HHV' (superior) o 'LHV' (inferior). No se supone.")
    hv = float(heating_value_J_mol)
    h_m, h_e = _pure_heating_values("CH4")[basis], _pure_heating_values("C2H6")[basis]
    warnings: list[str] = []
    if hv >= h_m:
        x = (hv - h_m) / (h_e - h_m)
        if x > FIT_REFUSE_ETHANE:
            raise ValueError(f"El poder calorífico medido ({hv / 1e3:.1f} kJ/mol, {basis}) es demasiado alto para un gas natural "
                             f"(equivale a {100 * x:.0f} % de etano). Revisa las condiciones base del volumen (temperatura y "
                             "presión del recibo), la base (superior o inferior) y que el combustible sea gas natural y no gas LP.")
        comp = {"CH4": 1.0 - x, "C2H6": x} if x > 0 else {"CH4": 1.0}
        if x > FIT_WARN_ETHANE:
            warnings.append(f"El poder calorífico del recibo equivale a {100 * x:.1f} % de etano, más rico que los gases de "
                            "referencia: revisa las condiciones base del volumen y la base del poder calorífico.")
        family = "metano + etano"
    else:
        y = 1.0 - hv / h_m
        if y > FIT_REFUSE_NITROGEN:
            raise ValueError(f"El poder calorífico medido ({hv / 1e3:.1f} kJ/mol, {basis}) es demasiado bajo para un gas natural "
                             f"(equivale a {100 * y:.0f} % de inertes). Revisa las condiciones base del volumen (temperatura y "
                             "presión del recibo) y la base (superior o inferior).")
        comp = {"CH4": 1.0 - y, "N2": y}
        if y > FIT_WARN_NITROGEN:
            warnings.append(f"El poder calorífico del recibo equivale a {100 * y:.1f} % de inertes, más pobre que casi todos los "
                            "gases de referencia: revisa las condiciones base del volumen y la base del poder calorífico.")
        family = "metano + nitrógeno"
    return {"composition": comp, "family": family, "warnings": warnings,
            "heating_value_check": {"measured_J_mol": hv, "fitted_J_mol": molar_heating_value(comp, basis), "basis": basis}}


def _measured_molar_heating_value(fuel: dict, fuel_cost: dict | None) -> dict | None:
    """The plant's heating value per mole and where it came from: the supplier's figure, or energy ÷ volume of the bill."""
    hv = fuel.get("heating_value")
    if hv is not None:
        extra = sorted(k for k in hv if k not in HEATING_VALUE_KEYS)
        if extra:
            raise ValueError(f"fuel.heating_value: claves desconocidas {extra}. Válidas: {HEATING_VALUE_KEYS}.")
        ref = hv.get("volume_reference") or {}
        if hv.get("value_J_m3") is None or hv.get("basis") not in ("HHV", "LHV") or ref.get("T_K") is None \
                or ref.get("P_Pa") is None:
            raise ValueError("fuel.heating_value: da value_J_m3 (poder calorífico por metro cúbico), basis 'HHV'|'LHV' y "
                             "volume_reference {'T_K', 'P_Pa'} (las condiciones base de ese metro cúbico). Vienen en el recibo "
                             "o en el contrato del gas; no se suponen.")
        mol_per_m3 = float(ref["P_Pa"]) / (CB.R_GAS * float(ref["T_K"]))
        return {"J_mol": float(hv["value_J_m3"]) / mol_per_m3, "basis": hv["basis"],
                "from": f"poder calorífico del proveedor, {float(hv['value_J_m3']) / 1e6:.3f} MJ/m³ a {ref['T_K']} K y "
                        f"{ref['P_Pa']} Pa (gas ideal)"}
    bill = (fuel_cost or {}).get("bill") or {}
    if bill.get("energy_GJ") is not None and bill.get("volume_m3") is not None:
        ref = bill.get("volume_reference") or {}
        if bill.get("basis") not in ("HHV", "LHV") or ref.get("T_K") is None or ref.get("P_Pa") is None:
            raise ValueError("fuel_cost.bill: para sacar el poder calorífico del recibo hacen falta basis ('HHV'|'LHV', la base "
                             "en la que se factura la energía) y volume_reference {'T_K', 'P_Pa'} (condiciones base del "
                             "volumen). Vienen en el recibo o en el contrato; no se suponen.")
        mol = float(ref["P_Pa"]) * float(bill["volume_m3"]) / (CB.R_GAS * float(ref["T_K"]))
        return {"J_mol": float(bill["energy_GJ"]) * 1e9 / mol, "basis": bill["basis"],
                "from": f"recibo del periodo «{bill.get('period')}»: {float(bill['energy_GJ']):,.2f} GJ ÷ "
                        f"{float(bill['volume_m3']):,.1f} m³ a {ref['T_K']} K y {ref['P_Pa']} Pa (gas ideal) = "
                        f"{float(bill['energy_GJ']) * 1e3 / float(bill['volume_m3']):.3f} MJ/m³"}
    return None


def resolve_natural_gas(fuel: dict, fuel_cost: dict | None) -> tuple[dict, dict]:
    """`{'type': 'natural_gas_unknown', ...}` -> (fuel dict for the combustion engine, how it was obtained)."""
    extra = sorted(k for k in fuel if k not in UNKNOWN_KEYS)
    if extra:
        raise ValueError(f"fuel (natural_gas_unknown): claves desconocidas {extra}. Válidas: {UNKNOWN_KEYS}.")
    if fuel.get("T_K") is None:
        raise ValueError("fuel.T_K: da la temperatura del gas combustible en el quemador (si no hay medición, la del cuarto de "
                         "calderas). No se supone.")
    measured = _measured_molar_heating_value(fuel, fuel_cost)
    if measured is not None:
        if fuel.get("reference_gas") is not None:
            raise ValueError("fuel: diste reference_gas y también un poder calorífico medido (fuel.heating_value o energía y "
                             "volumen en el recibo). Con poder calorífico medido la composición se ajusta a él: quita "
                             "reference_gas.")
        fit = fit_to_heating_value(measured["J_mol"], measured["basis"])
        info = {"method": "fit", "composition": fit["composition"], "family": fit["family"], "measured": measured,
                "heating_value_check": fit["heating_value_check"], "warnings": fit["warnings"],
                "description": f"mezcla equivalente de {fit['family']} ajustada para dar exactamente el poder calorífico medido "
                               f"({measured['J_mol'] / 1e3:.1f} kJ/mol, base {'PCS' if measured['basis'] == 'HHV' else 'PCI'}; "
                               f"{measured['from']})"}
    else:
        name = fuel.get("reference_gas") or DEFAULT_REFERENCE_GAS
        ref = reference_gas(name)
        info = {"method": "reference", "composition": dict(ref["composition"]), "reference_gas": name,
                "reference_was_named": fuel.get("reference_gas") is not None, "source": ref["source"],
                "notes": list(ref["notes"]), "warnings": [],
                "description": f"gas natural de referencia «{name}» ({ref['source']}), porque la planta no dio composición ni "
                               "poder calorífico"}
    return {"type": "gas", "composition": dict(info["composition"]), "T_K": float(fuel["T_K"])}, info


def surrogate_error(name: str, flue_O2_dry: float = 0.04, stack_T_K: float = 450.0, air_T_K: float = 298.15,
                    pressure_Pa: float = 101325.0) -> dict:
    """Full composition of a reference gas against its fitted two-component surrogate, through the same engine.

    Arguments are the comparison point (not plant data): one measured O2 and one stack temperature for both gases.
    """
    full = reference_gas(name)["composition"]
    fit = fit_to_heating_value(molar_heating_value(full, "HHV"), "HHV")
    out = {}
    for label, comp in (("full", full), ("surrogate", fit["composition"])):
        r = CB.combustion_efficiency({"type": "gas", "composition": comp, "T_K": air_T_K}, {"T_K": air_T_K, "relative_humidity": 0.0},
                                     pressure_Pa, flue_O2={"mole_fraction": flue_O2_dry, "basis": "dry"}, stack_T_K=stack_T_K,
                                     setting_loss_fraction=0.0, setting_loss_basis="HHV", adiabatic_flame=False)
        wet = r["flue_gas"]["wet_mole_fractions"]
        co2_kg_per_GJ = r["flue_gas"]["mol_per_kg_fuel"] * wet.get("CO2", 0.0) * 0.0440095 / (r["fuel"]["HHV_J_kg"] / 1e9)
        out[label] = {"efficiency_HHV": r["efficiency"]["HHV"]["efficiency"], "efficiency_LHV": r["efficiency"]["LHV"]["efficiency"],
                      "excess_air_fraction": r["air"]["excess_air_fraction"], "co2_kg_per_GJ_HHV": co2_kg_per_GJ,
                      "water_dew_point_K": r["flue_gas"]["water_dew_point_K"]}
    f, s = out["full"], out["surrogate"]
    return {"gas": name, "surrogate_composition": fit["composition"], "full": f, "surrogate": s,
            "efficiency_HHV_difference_points": 100.0 * (s["efficiency_HHV"] - f["efficiency_HHV"]),
            "efficiency_LHV_difference_points": 100.0 * (s["efficiency_LHV"] - f["efficiency_LHV"]),
            "excess_air_difference_points": 100.0 * (s["excess_air_fraction"] - f["excess_air_fraction"]),
            "co2_difference_fraction": s["co2_kg_per_GJ_HHV"] / f["co2_kg_per_GJ_HHV"] - 1.0,
            "dew_point_difference_K": s["water_dew_point_K"] - f["water_dew_point_K"]}
