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
    fn = {"heat_exchanger": sslw.cost_heat_exchanger_floored, "pump": sslw.cost_pump, "compressor": sslw.cost_compressor,
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



def flowsheet_cost(flowsheet_id: str, prices: dict, hours_per_year: float, interest: float, years: int,
                   exchanger_areas_m2: dict | None = None, vessel_sizes: dict | None = None, options: dict | None = None,
                   lang: str | float = "lang_1948_fluid", country: str | None = None, year: int = 2023,
                   cepci: float | None = None, exchanger_specs: dict | None = None) -> dict:
    """Total annual cost (USD/y) of a SOLVED flowsheet = annualized fixed capital + operating cost.
    Capital: pumps and compressors are sized from the simulation; heaters, coolers and heat exchangers need their area
    in exchanger_areas_m2 {unit: m2} (compute it with heat_exchanger_area); vessels need vessel_sizes {unit: {diameter_m,
    length_m}}; units without a size are listed in `not_costed`. Fixed capital = Σ purchase × Lang factor × location
    factor; lang: lang_1948_fluid (4.74, default) | lang_1948_solid (3.10) | peters_2003_fluid_fci (5.0) |
    peters_2003_fluid_tci (6.0) | a number. country: IDAES location factor (e.g. "Mexico" → mid of 0.84-0.92).
    Operating: pump/compressor electricity, heater duty × heating price, cooler duty × cooling price.
    prices (required for what the flowsheet uses): electricity_usd_kWh, heating_usd_GJ, cooling_usd_GJ.
    Annualized capital = FCI × i(1+i)^n/((1+i)^n − 1) with interest (fraction) and years.
    exchanger_specs {unit: {U_W_m2K, utility_in_K, utility_out_K, shell_passes}} sizes exchangers automatically from the
    solved duty and stream temperatures (heater: utility is the hot side, e.g. condensing steam in = out; cooler: utility
    is the cold side; heat_exchanger: only U, area = UA/U). Use it inside optimize_design, where duties change."""
    from packages.costing.annual import annual_cost, location_factor
    s = _s(flowsheet_id)
    opts = options or {}
    areas, vessels = dict(exchanger_areas_m2 or {}), vessel_sizes or {}
    sized = _auto_areas(flowsheet_id, exchanger_specs or {}, areas)
    purchase, energy, not_costed, notes = {}, {}, [], {}
    for unit, kind in s["kinds"].items():
        u = read_unit(flowsheet_id, unit)
        if kind in ("pump", "compressor"):
            c = cost_unit(flowsheet_id, unit, opts.get(unit), year, cepci)
            purchase[unit] = c["purchase_cost_usd"]
            energy[unit] = ("electricity", float(u.get("power_kW") or u.get("duty_or_power_kW") or 0.0))
        elif kind in ("heater", "cooler", "heat_exchanger"):
            if kind == "heater":
                energy[unit] = ("heating", float(u.get("duty_or_power_kW") or 0.0))
            elif kind == "cooler":
                energy[unit] = ("cooling", float(u.get("heat_removed_kW") or u.get("duty_or_power_kW") or 0.0))
            if unit in areas and areas[unit] <= 0.0:
                pass                                   # duty ≈ 0: no exchanger to buy
            elif unit in areas:
                purchase[unit] = equipment_cost("heat_exchanger", {"area_m2": float(areas[unit])}, opts.get(unit), year, cepci)["purchase_cost_usd"]
            else:
                not_costed.append(unit)
        elif kind == "separator" or unit in vessels:
            if unit in vessels:
                purchase[unit] = equipment_cost("vessel", dict(vessels[unit]), opts.get(unit), year, cepci)["purchase_cost_usd"]
            else:
                not_costed.append(unit)
        elif kind not in ("mixer", "splitter", "valve", "recycle", "pipe"):
            not_costed.append(unit)
            notes[unit] = f"{kind}: no cost correlation exposed yet"
    loc = location_factor(country) if country else 1.0
    out = annual_cost(purchase, energy, prices, hours_per_year, interest, years, lang, loc)
    return out | {"not_costed": not_costed, "notes": notes, "year": year, "areas_m2": areas, "auto_sized": sized,
                  "warning": "Estimado de orden de magnitud: compra de Seider vía IDAES y capital por factor de Lang" if purchase else
                  "Sin equipos costeados: solo costo de operación"}


def _auto_areas(flowsheet_id: str, specs: dict, areas: dict) -> dict:
    """Fill `areas` for units in specs from the solved flowsheet; returns {unit: {area_m2, LMTD_K, F}}."""
    from packages.transport.hydraulics import hx_area
    s = _s(flowsheet_id)
    done = {}
    for unit, sp in specs.items():
        if unit in areas:
            continue
        kind = s["kinds"][unit]
        U = float(sp["U_W_m2K"])
        if kind == "heat_exchanger" and "UA_W_K" in s["params"].get(unit, {}):
            areas[unit] = float(s["params"][unit]["UA_W_K"]) / U
            done[unit] = {"area_m2": areas[unit], "from": "UA/U"}
            continue
        c = s["conn"][unit]
        tin = s["streams"][c["feeds"][0]].TemperatureK
        tout = s["streams"][c["products"][0]].TemperatureK
        u = read_unit(flowsheet_id, unit)
        q = abs(float(u.get("heat_removed_kW") if kind == "cooler" else u.get("duty_or_power_kW") or 0.0))
        if q < 1e-3:
            done[unit] = {"area_m2": 0.0, "note": "duty ≈ 0: no exchanger needed"}
            continue
        ui, uo = float(sp["utility_in_K"]), float(sp.get("utility_out_K", sp["utility_in_K"]))
        if kind == "heater":
            r = hx_area(q * 1e3, ui, uo, tin, tout, U, int(sp.get("shell_passes", 0)))
        elif kind == "cooler":
            r = hx_area(q * 1e3, tin, tout, ui, uo, U, int(sp.get("shell_passes", 0)))
        else:
            raise ValueError(f"{unit}: exchanger_specs for {kind} needs UA_W_K in its params")
        areas[unit] = r["area_m2"]
        done[unit] = {"area_m2": r["area_m2"], "LMTD_K": r["LMTD_K"], "F": r["F"]}
    return done


COSTING_TOOLS = [equipment_cost, costing_options, cost_unit, flowsheet_cost]
