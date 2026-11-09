"""Agent tools for plant economics (bloque M3): bare-module capital, total capital investment and discounted cash flow.
No DWSIM flowsheet needed; purchase costs come from equipment_cost / cost_unit. Engine: `packages.costing.economics`."""
from __future__ import annotations

from packages.costing import economics as EC


def bare_module_factors() -> dict:
    """Bare-module factors F_BM by equipment type (Seider et al. 2017, read from BioSTEAM), the keys capital_investment
    accepts as `type`, and their source class."""
    return EC.bare_module_factors()


def capital_investment(equipment: list[dict], factors: dict, location_factor: float = 1.0) -> dict:
    """Installed (bare-module) cost and total capital investment. equipment: [{name, type (key of bare_module_factors,
    e.g. heat_exchanger_Utube, pump, compressor, vessel_vertical, tower) or F_BM (yours, for anything not listed),
    purchase_cost_usd, base_purchase_cost_usd and platforms_ladders_purchase_cost_usd (both returned by equipment_cost),
    basis}]. C_BM = C_P + C_P,base·(F_BM − 1). factors (fractions, ALL required, 0 if it does not apply; they are the
    user's): site_preparation, service_facilities, allocated_costs (of C_TBM); contingency_and_fee (of C_DPI); land,
    royalties, startup (of C_TDC); working_capital (of C_TPI). location_factor multiplies C_TBM (Mexico 0.84-0.92 in
    IDAES's table). Returns every level (C_TBM, C_DPI, C_TDC, C_TPI, TCI), a Lang cross-check and evidence."""
    return EC.capital_investment(equipment, factors, location_factor)


def cash_flow_analysis(depreciable_capital_usd: float, revenue_usd_y: float, operating_cost_usd_y: float,
                       plant_life_years: int, income_tax_rate: float, discount_rate: float, depreciation: str,
                       construction_schedule: list[float], working_capital_usd: float = 0.0, land_usd: float = 0.0,
                       startup_years: float = 0.0, startup_revenue_fraction: float = 1.0,
                       startup_operating_cost_fraction: float = 1.0) -> dict:
    """Discounted cash flow in real terms, no loans: NPV, IRR, payback, discounted payback, break-even revenue and the
    yearly table. depreciable_capital_usd = C_TDC from capital_investment; operating_cost_usd_y = cash costs WITHOUT
    depreciation; rates as fractions (0.30); depreciation 'MACRS3|5|7|10|15|20' (IRS, U.S.), 'SL<n>' (straight line;
    Mexico's ISR rates, e.g. SL10), 'DDB<n>', 'SYD<n>'; construction_schedule fractions per year summing to 1 ([0.4,
    0.6]). Working capital is spent in the last construction year and recovered at the end; land likewise from year
    one. All economic inputs are the user's: never assume prices, tax or discount rate."""
    return EC.cash_flow_analysis(depreciable_capital_usd, revenue_usd_y, operating_cost_usd_y, plant_life_years,
                                 income_tax_rate, discount_rate, depreciation, construction_schedule, working_capital_usd,
                                 land_usd, startup_years, startup_revenue_fraction, startup_operating_cost_fraction)


ECONOMICS_TOOLS = [bare_module_factors, capital_investment, cash_flow_analysis]
