"""Heat-recovery trade-off (bloque 20): the first cost optimization, all on deterministic tools.

Cold feed (water, 10 kg/s, 300 K) must reach 395 K; hot product (water, 10 kg/s, 400 K) must leave at 305 K
(full recovery would need a 5 K approach, ~940 m², so the optimum is interior). A
feed/product exchanger E-1 of area A recovers heat; a steam heater H-1 and a cooling-water cooler C-1 make up the rest.
Bigger A → more capital, less steam and cooling water. For each A: DWSIM solves the flowsheet (E-1 in CalcBothTemp_UA),
heat_exchanger_area sizes H-1 (steam condensing at 450 K) and C-1 (cooling water 290 → 300 K), flowsheet_cost
prices everything (Seider via IDAES, Lang, CRF) → total annual cost. Prices are ILLUSTRATIVE (MOCKS.md).
"""
from __future__ import annotations

ILLUSTRATIVE_PRICES = {"heating_usd_GJ": 8.0, "cooling_usd_GJ": 0.5, "electricity_usd_kWh": 0.10}
U = {"E-1": 850.0, "H-1": 1100.0, "C-1": 850.0}           # W/m²K, water services (illustrative design values)
STEAM_K, CW_IN, CW_OUT = 450.0, 290.0, 300.0
T_FEED_OUT, T_PROD_OUT = 395.0, 305.0


def evaluate(area_m2: float, prices: dict | None = None, hours: float = 8000.0, interest: float = 0.10, years: int = 10) -> dict:
    from packages.dwsim_client import tools as T           # loads DWSIM / .NET first
    from DWSIM.UnitOperations.UnitOperations import HeatExchangerCalcMode
    from packages.dwsim_client.tools_costing import flowsheet_cost
    from packages.transport.hydraulics import hx_area
    prices = prices or ILLUSTRATIVE_PRICES
    fid = T.new_flowsheet(f"hr_{area_m2:.1f}", ["Water"], "SteamTables")["flowsheet_id"]
    try:
        T.add_material_stream(fid, "feed", temperature_K=300.0, pressure_Pa=4e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
        T.add_material_stream(fid, "prod", temperature_K=400.0, pressure_Pa=4e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
        for n in ("feed_pre", "prod_pre", "feed_out", "prod_out"):
            T.add_material_stream(fid, n)
        s = T._s(fid)
        b = (s["fs"].AddHeatExchanger("E-1").WithCalculationMode(HeatExchangerCalcMode.CalcBothTemp_UA)
             .WithGlobalUA(U["E-1"] * area_m2).WithExchangeArea(area_m2))
        s["units"]["E-1"], s["kinds"]["E-1"], s["params"]["E-1"] = b, "heat_exchanger", {"UA_W_K": U["E-1"] * area_m2}
        T.connect(fid, "E-1", feeds=["prod", "feed"], products=["prod_pre", "feed_pre"])
        T.add_unit(fid, "heater", "H-1", {"outlet_T_K": T_FEED_OUT})
        T.connect(fid, "H-1", feeds=["feed_pre"], products=["feed_out"])
        T.add_unit(fid, "cooler", "C-1", {"outlet_T_K": T_PROD_OUT})
        T.connect(fid, "C-1", feeds=["prod_pre"], products=["prod_out"])
        r = T.solve(fid)
        if not r["solved"]:
            raise RuntimeError(r.get("solver_errors"))
        st = r["streams"]
        q_rec = abs(float(getattr(b.Object, "__implementation__", b.Object).Q))
        q_h = abs(T.read_unit(fid, "H-1")["duty_or_power_kW"])
        q_c = abs(T.read_unit(fid, "C-1")["heat_removed_kW"])
        areas = {"E-1": area_m2}
        if q_h > 1.0:
            areas["H-1"] = hx_area(q_h * 1e3, STEAM_K, STEAM_K, st["feed_pre"]["T_K"], T_FEED_OUT, U["H-1"])["area_m2"]
        if q_c > 1.0:
            areas["C-1"] = hx_area(q_c * 1e3, st["prod_pre"]["T_K"], T_PROD_OUT, CW_IN, CW_OUT, U["C-1"])["area_m2"]
        cost = flowsheet_cost(fid, prices, hours, interest, years, exchanger_areas_m2=areas,
                              options={k: {"material": "CarbonSteelCarbonSteel"} for k in areas})
        return {"area_E1_m2": area_m2, "Q_recovered_kW": q_rec, "Q_steam_kW": q_h, "Q_cooling_kW": q_c,
                "approach_K": st["prod_pre"]["T_K"] - 300.0, "T_feed_pre_K": st["feed_pre"]["T_K"], "areas_m2": areas,
                "TAC_usd_y": cost["total_annual_cost_usd_y"], "capital_usd_y": cost["annualized_capital_usd_y"],
                "operating_usd_y": cost["operating_cost_usd_y"], "fixed_capital_usd": cost["fixed_capital_usd"],
                "not_costed": cost["not_costed"], "energy_rel_err": max(u["energy_rel_err"] for u in r["units"])}
    finally:
        T.close_flowsheet(fid)


def optimize(lo: float = 10.0, hi: float = 900.0, n: int = 14, refine: int = 12, **kw) -> dict:
    """Log-spaced scan, then golden-section on the bracket around the best point (TAC is unimodal in A here)."""
    import math
    grid = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    scan = [evaluate(a, **kw) for a in grid]
    k = min(range(n), key=lambda i: scan[i]["TAC_usd_y"])
    a, b = grid[max(0, k - 1)], grid[min(n - 1, k + 1)]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = evaluate(c, **kw), evaluate(d, **kw)
    for _ in range(refine):
        if fc["TAC_usd_y"] < fd["TAC_usd_y"]:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = evaluate(c, **kw)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = evaluate(d, **kw)
    best = fc if fc["TAC_usd_y"] < fd["TAC_usd_y"] else fd
    return {"scan": scan, "best": best, "bracket": [a, b]}
