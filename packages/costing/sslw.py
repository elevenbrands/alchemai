"""Equipment purchase cost with IDAES's implementation of Seider, Seader, Lewin & Widagdo («Product and Process Design
Principles», 3rd ed.), used verbatim: a placeholder unit block carries the sizes, IDAES builds its own costing
constraints, and Pyomo's strongly-connected-component solve evaluates them (each is explicit in one variable, so no
external NLP solver is needed). The same numbers as IDAES's own tests are reproduced in tests/test_costing.py.

Costs are PURCHASE costs (f.o.b.), not installed: multiply by a bare-module or Lang factor for that. Currency:
IDAES's CEPCI table (USD_CE500 base) covers 1990-2023; for a later year pass `cepci` (the user's index value).
"""
from __future__ import annotations

import logging
from typing import Any

logging.getLogger("idaes").setLevel(logging.ERROR)
logging.getLogger("pyomo").setLevel(logging.ERROR)

CEPCI_IDAES = {2015: 556.8, 2016: 541.7, 2017: 567.5, 2018: 603.1, 2019: 607.5, 2020: 596.2, 2021: 708.0, 2022: 816.0, 2023: 797.9}


def _enum(cls, name: str):
    try:
        return cls[name]
    except KeyError:
        raise ValueError(f"{name!r} not in {cls.__name__}: {[m.name for m in cls]}") from None


def _run(build) -> dict:
    from idaes.core import FlowsheetBlock, UnitModelBlock
    from idaes.core.util.model_statistics import degrees_of_freedom
    from idaes.models.costing.SSLW import SSLWCosting
    from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
    from pyomo.environ import ConcreteModel, Var, units as u, value
    m = ConcreteModel()
    m.fs = FlowsheetBlock()
    m.fs.costing = SSLWCosting()
    m.fs.unit = UnitModelBlock()
    build(m)
    dof = degrees_of_freedom(m)
    if dof != 0:
        raise RuntimeError(f"costing model has {dof} degrees of freedom")
    solve_strongly_connected_components(m)
    c = m.fs.unit.costing
    detail = {v.local_name: float(value(v)) for v in c.component_objects(Var, descend_into=False)
              if v.local_name not in ("capital_cost",) and not v.is_indexed()}
    return {"usd_ce500": float(value(c.capital_cost)), "detail": detail,
            "_convert": lambda year: float(value(u.convert(c.capital_cost, to_units=getattr(u, f"USD_{year}"))))}


def _price(r: dict, year: int, cepci: float | None) -> dict:
    if cepci is not None:
        usd = r["usd_ce500"] * cepci / 500.0
        basis = f"CEPCI {cepci} (dado por el usuario)"
    else:
        if year not in CEPCI_IDAES:
            raise ValueError(f"IDAES has CEPCI for 2015-2023 only; for {year} pass cepci=<index value>")
        usd = r["_convert"](year)
        basis = f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)"
    return {"purchase_cost_usd": usd, "year": year, "basis": basis, "usd_ce500": r["usd_ce500"],
            "detail": r["detail"], "method": "Seider et al. (SSLW) vía IDAES; costo de compra, no instalado"}


def cost_heat_exchanger(area_m2: float, shell_pressure_Pa: float = 2 * 101325, hx_type: str = "Utube",
                        material: str = "StainlessSteelStainlessSteel", tube_length: str = "TwelveFoot",
                        year: int = 2023, cepci: float | None = None) -> dict:
    from idaes.models.costing.SSLW import HXMaterial, HXTubeLength, HXType, SSLWCostingData
    from idaes.core import UnitModelCostingBlock
    from pyomo.environ import Block, Param, units as u

    def build(m):
        m.fs.unit.area = Param(initialize=area_m2, units=u.m ** 2)
        m.fs.unit.hot_side = Block()
        m.fs.unit.hot_side.properties_in = Block(m.fs.time)
        m.fs.unit.hot_side.properties_in[0].pressure = Param(initialize=shell_pressure_Pa, units=u.Pa)
        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, costing_method=SSLWCostingData.cost_heat_exchanger,
                                                  costing_method_arguments={"hx_type": _enum(HXType, hx_type), "material_type": _enum(HXMaterial, material),
                                                                            "tube_length": _enum(HXTubeLength, tube_length)})
    return _price(_run(build), year, cepci) | {"equipment": "heat_exchanger", "area_m2": area_m2}


def cost_heat_exchanger_floored(area_m2: float, **kw) -> dict:
    """Seider's correlation is quadratic in ln(area): below its minimum (≈10 m² for carbon steel) it rises again and a
    0.1 m² exchanger would cost more than a 40 m² one. Below the minimum, price at the minimum and say so."""
    import math
    key = tuple(sorted((k, v) for k, v in kw.items() if k not in ("year", "cepci")))
    a_min = _FLOOR.get(key)
    if a_min is None:
        lo, hi = math.log(1.0), math.log(100.0)
        g = (math.sqrt(5) - 1) / 2
        f = lambda la: cost_heat_exchanger(math.exp(la), **kw)["usd_ce500"]
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        fc, fd = f(c), f(d)
        for _ in range(30):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - g * (hi - lo); fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + g * (hi - lo); fd = f(d)
        a_min = _FLOOR[key] = math.exp(0.5 * (lo + hi))
    if area_m2 >= a_min:
        return cost_heat_exchanger(area_m2, **kw)
    out = cost_heat_exchanger(a_min, **kw) | {"area_m2": area_m2, "priced_at_area_m2": a_min}
    out["warning"] = (f"Área {area_m2:.2f} m² bajo el mínimo de la correlación ({a_min:.1f} m²): se cotiza al mínimo; "
                      "un equipo tan chico suele ser de doble tubo o de placas, no de coraza y tubos")
    return out


_FLOOR: dict = {}


def cost_pump(power_W: float, deltaP_Pa: float, density_kg_m3: float, flow_m3_s: float, pump_type: str = "Centrifugal",
              material: str = "CastIron", pump_type_factor: float = 1.4, motor_type: str = "Open", year: int = 2023,
              cepci: float | None = None) -> dict:
    """pump_type_factor (Seider table): 1.0, 1.5, 2.0 or 1.4 for the single/two-stage, speed and head classes."""
    from idaes.models.costing.SSLW import PumpMaterial, PumpMotorType, PumpType, SSLWCostingData
    from idaes.core import UnitModelCostingBlock
    from pyomo.common.config import ConfigValue
    from pyomo.environ import Block, Param, units as u

    def build(m):
        m.fs.unit.config.declare("compressor", ConfigValue(default=True))
        m.fs.unit.work_mechanical = Param([0], initialize=power_W, units=u.W)
        m.fs.unit.deltaP = Param([0], initialize=deltaP_Pa, units=u.Pa)
        m.fs.unit.control_volume = Block()
        m.fs.unit.control_volume.properties_in = Block(m.fs.time)
        m.fs.unit.control_volume.properties_in[0].dens_mass = Param(initialize=density_kg_m3, units=u.kg / u.m ** 3)
        m.fs.unit.control_volume.properties_in[0].flow_vol = Param(initialize=flow_m3_s, units=u.m ** 3 / u.s)
        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, costing_method=SSLWCostingData.cost_pump,
                                                  costing_method_arguments={"pump_type": _enum(PumpType, pump_type), "material_type": _enum(PumpMaterial, material),
                                                                            "pump_type_factor": pump_type_factor, "motor_type": _enum(PumpMotorType, motor_type)})
    return _price(_run(build), year, cepci) | {"equipment": "pump", "power_kW": power_W / 1e3}


def cost_compressor(power_W: float, compressor_type: str = "Centrifugal", drive_type: str = "ElectricMotor",
                    material: str = "CarbonSteel", year: int = 2023, cepci: float | None = None) -> dict:
    from idaes.models.costing.SSLW import CompressorDriveType, CompressorMaterial, CompressorType, SSLWCostingData
    from idaes.core import UnitModelCostingBlock
    from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
    from pyomo.common.config import ConfigValue
    from pyomo.environ import Param, units as u

    def build(m):
        m.fs.unit.config.declare("compressor", ConfigValue(default=True))
        m.fs.unit.config.declare("thermodynamic_assumption", ConfigValue(default=ThermodynamicAssumption.isentropic))
        m.fs.unit.work_mechanical = Param([0], initialize=power_W, units=u.W)
        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, costing_method=SSLWCostingData.cost_compressor,
                                                  costing_method_arguments={"compressor_type": _enum(CompressorType, compressor_type),
                                                                            "drive_type": _enum(CompressorDriveType, drive_type),
                                                                            "material_type": _enum(CompressorMaterial, material)})
    return _price(_run(build), year, cepci) | {"equipment": "compressor", "power_kW": power_W / 1e3}


def cost_vessel(diameter_m: float, length_m: float, vertical: bool = True, material: str = "CarbonSteel",
                include_platforms_ladders: bool = True, weight_limit: int = 1, aspect_ratio_range: int = 1,
                number_of_trays: int | None = None, tray_type: str = "Sieve", tray_material: str = "CarbonSteel",
                year: int = 2023, cepci: float | None = None) -> dict:
    """Pressure vessel shell (Seider): weight from diameter, length and the default shell thickness of IDAES.
    number_of_trays (vertical only) adds Seider's tray cost: C_BT = 468·exp(0.1739·D_ft) per tray × F_NT (= 2.25/1.0414^N,
    at least 1) × F_TT (Sieve 1, Valve 1.18, BubbleCap 1.87) × F_TM (material, grows with D); valid for D of 2-16 ft."""
    from idaes.models.costing.SSLW import SSLWCostingData, TrayMaterial, TrayType, VesselMaterial
    from idaes.core import UnitModelCostingBlock
    from pyomo.environ import Param, units as u

    def build(m):
        m.fs.unit.length = Param(initialize=length_m, units=u.m)
        m.fs.unit.diameter = Param(initialize=diameter_m, units=u.m)
        args = {"vertical": vertical, "material_type": _enum(VesselMaterial, material), "weight_limit": weight_limit,
                "aspect_ratio_range": aspect_ratio_range, "include_platforms_ladders": include_platforms_ladders}
        if number_of_trays is not None:
            args |= {"number_of_trays": int(number_of_trays), "tray_type": _enum(TrayType, tray_type), "tray_material": _enum(TrayMaterial, tray_material)}
        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, costing_method=SSLWCostingData.cost_vessel,
                                                  costing_method_arguments=args)

    r = _run(build)
    out = _price(r, year, cepci) | {"equipment": "vessel", "diameter_m": diameter_m, "length_m": length_m}
    if number_of_trays is not None:
        out |= {"number_of_trays": int(number_of_trays), "tray_type": tray_type,
                "trays_purchase_cost_usd": r["detail"]["base_cost_trays"] * out["purchase_cost_usd"] / r["usd_ce500"]}
    return out


def cost_packing(volume_m3: float, price_usd_per_m3: float, year: int = 2023, cepci: float | None = None) -> dict:
    """Packing purchase cost = volume × unit price. There is no verified public price table (vendor quotes, strongly
    dependent on material and size), so the unit price is an INPUT and the result says so; the CEPCI scaling uses the
    year the price refers to (`year`)."""
    return {"equipment": "packing", "volume_m3": volume_m3, "price_usd_per_m3": price_usd_per_m3,
            "purchase_cost_usd": volume_m3 * price_usd_per_m3, "year": year,
            "note": "precio unitario del relleno = dato de entrada del usuario (sin tabla pública verificada)"}


def options() -> dict[str, Any]:
    from idaes.models.costing import SSLW as S
    return {k: [m.name for m in getattr(S, k)] for k in ("HXType", "HXMaterial", "HXTubeLength", "PumpType", "PumpMaterial", "PumpMotorType",
                                                        "CompressorType", "CompressorDriveType", "CompressorMaterial", "VesselMaterial")}
