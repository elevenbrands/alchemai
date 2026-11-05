"""Agent tools for equipment purchase cost (Seider et al. via IDAES, see packages/costing/sslw.py).

`equipment_cost` prices a size given by hand or by another tool (heat_exchanger_area, size_pipe, ...);
`cost_unit` reads the size of a solved DWSIM pump or compressor (power, ΔP, inlet density and flow) and prices it."""
from __future__ import annotations

from packages.costing import sslw
from packages.dwsim_client.tools import _s, read_unit


def equipment_cost(equipment: str, size: dict, options: dict | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Purchase cost (USD, f.o.b., NOT installed) by Seider et al. as implemented in IDAES. equipment and size keys:
      heat_exchanger {area_m2, shell_pressure_Pa}; pump {power_W, deltaP_Pa, density_kg_m3, flow_m3_s};
      compressor {power_W}; vessel {diameter_m, length_m, vertical (bool)}.
    options (optional, names exactly as listed by costing_options): heat_exchanger hx_type (default Utube), material
    (StainlessSteelStainlessSteel), tube_length (TwelveFoot); pump pump_type (Centrifugal), material (CastIron),
    pump_type_factor (1.4), motor_type (Open); compressor compressor_type (Centrifugal), drive_type (ElectricMotor),
    material (CarbonSteel); vessel material (CarbonSteel), include_platforms_ladders (true).
    year: 2015-2023 uses IDAES's CEPCI table; for another year pass cepci (the index value)."""
    fn = {"heat_exchanger": sslw.cost_heat_exchanger, "pump": sslw.cost_pump, "compressor": sslw.cost_compressor,
          "vessel": sslw.cost_vessel}.get(equipment)
    if fn is None:
        raise ValueError("equipment must be heat_exchanger, pump, compressor or vessel")
    return fn(**size, **(options or {}), year=year, cepci=cepci)


def costing_options() -> dict:
    """Valid option names for equipment_cost (types, materials, motors, tube lengths)."""
    return sslw.options()


def cost_unit(flowsheet_id: str, unit: str, options: dict | None = None, year: int = 2023, cepci: float | None = None) -> dict:
    """Purchase cost of a solved DWSIM pump or compressor, sized from the simulation (power; for pumps also ΔP and
    the inlet density and volumetric flow). For heat exchangers compute the area first (heat_exchanger_area) and
    call equipment_cost; for vessels give diameter and length."""
    s = _s(flowsheet_id)
    kind = s["kinds"][unit]
    u = read_unit(flowsheet_id, unit)
    if kind == "compressor":
        p_kW = u.get("duty_or_power_kW")
        return equipment_cost("compressor", {"power_W": abs(float(p_kW)) * 1e3}, options, year, cepci) | {"unit": unit}
    if kind == "pump":
        from packages.dwsim_client.tools_transport import _phase_props
        b = s["units"][unit]
        o = getattr(b.Object, "__implementation__", b.Object)
        feed = next(n for n, st in s["streams"].items()
                    if any(getattr(c, "AttachedConnector", None) is not None and c.IsAttached and c.AttachedConnector.AttachedFrom.Name == st.Object.GraphicObject.Name
                           for c in o.GraphicObject.InputConnectors))
        pin = _phase_props(flowsheet_id, feed)
        dP = float(o.DeltaP) if getattr(o, "DeltaP", None) is not None else None
        size = {"power_W": abs(float(u.get("power_kW") or u.get("duty_or_power_kW"))) * 1e3, "deltaP_Pa": dP,
                "density_kg_m3": pin["density_kg_m3"], "flow_m3_s": pin["mass_flow_kg_s"] / pin["density_kg_m3"]}
        return equipment_cost("pump", size, options, year, cepci) | {"unit": unit, "size_from_simulation": size}
    raise ValueError(f"cost_unit sizes pumps and compressors; for {kind} use equipment_cost with the size")


COSTING_TOOLS = [equipment_cost, costing_options, cost_unit]
