"""Agent-facing wrappers of the utilities-capital tools (`packages.dwsim_client.tools_utilities_capital`)."""
from __future__ import annotations

import json

from anthropic import beta_tool

from packages.dwsim_client import tools_utilities_capital as TUC


@beta_tool
def cooling_tower_design(duty_W: float, supply_T_K: float, return_T_K: float, wet_bulb_T_K: float,
                         cycles_of_concentration: float, dry_bulb_T_K: float | None = None,
                         liquid_to_gas_ratio: float | None = None, air_outlet_T_K: float | None = None,
                         drift_fraction: float | None = None, pressure_Pa: float | None = None,
                         pump_head_m: float | None = None, pump_efficiency: float | None = None,
                         fan_static_pressure_Pa: float | None = None, fan_efficiency: float | None = None,
                         year: int = 2023, cepci: float | None = None) -> str:
    """Cooling tower design and CAPITAL from the heat duty, cold (supply) and hot (return) water temperatures, design
    wet-bulb and cycles of concentration (all the user's). Returns circulation, evaporation, drift, blowdown, make-up, fan
    and pump power, and capital.items (tower + pumps, ready for capital_investment) with purchase and installed cost and
    the spread between published sources. Optional: dry_bulb_T_K, site pressure_Pa, liquid_to_gas_ratio or
    air_outlet_T_K (gives design air flow and Merkel number), pump head/efficiency and fan static pressure/efficiency.
    Drift and reference powers are SUGGESTED from NREL (Humbird 2011) when omitted and flagged in suggested_inputs.
    year 2015-2023 or cepci for another year. Operating cost is utility_costs, not this tool."""
    return json.dumps(TUC.cooling_tower_design(duty_W, supply_T_K, return_T_K, wet_bulb_T_K, cycles_of_concentration,
                                               dry_bulb_T_K, liquid_to_gas_ratio, air_outlet_T_K, drift_fraction, pressure_Pa,
                                               pump_head_m, pump_efficiency, fan_static_pressure_Pa, fan_efficiency, year,
                                               cepci), default=float, ensure_ascii=False)


@beta_tool
def steam_boiler_design(steam_demand: list[dict[str, object]], feedwater_T_K: float, fuel: dict[str, object],
                        boiler: dict[str, object] | None = None, combustion: dict[str, object] | None = None,
                        turbogenerator: dict[str, object] | None = None, year: int = 2023,
                        cepci: float | None = None) -> str:
    """Steam boiler (and optional turbogenerator) sizing and CAPITAL. steam_demand [{name, pressure_Pa (abs), flow_kg_s}];
    fuel {type: natural_gas | fuel_oil | biomass, LHV_J_kg and/or HHV_J_kg as fired, or for biomass HHV_dry_J_kg +
    moisture_mass_fraction (+ hydrogen_dry_mass_fraction for LHV)}; boiler {efficiency, efficiency_basis HHV|LHV,
    superheat_K or steam_T_K (no turbine), pressure_Pa, blowdown_fraction} OR combustion {arguments of
    combustion_efficiency} to compute the efficiency; turbogenerator {inlet_pressure_Pa, inlet_T_K, isentropic_efficiency
    (the user's), mode back_pressure | condensing (+ condensing_flow_kg_s, condenser_pressure_Pa), generator_efficiency}.
    Returns duty, feedwater, blowdown, fuel rate, steam condition at each level, electric power and capital.items for
    capital_investment (published tables/quotes by fuel type, with the spread between sources)."""
    return json.dumps(TUC.steam_boiler_design(steam_demand, feedwater_T_K, fuel, boiler, combustion, turbogenerator, year,
                                              cepci), default=float, ensure_ascii=False)


@beta_tool
def wastewater_treatment_cost(flow_m3_s: float, cod_kg_m3: float, configuration: str, cod_removal: dict[str, float],
                              include_reverse_osmosis: bool = False, include_brine_evaporator: bool = False,
                              density_kg_m3: float | None = None, electricity_price_usd_kWh: float | None = None,
                              operating_hours_per_year: float | None = None, aeration_kg_O2_per_kWh: float | None = None,
                              biogas_value_usd_GJ: float | None = None, year: int = 2023, cepci: float | None = None) -> str:
    """Conventional wastewater treatment: COD balance, biogas methane, oxygen demand, installed capital (ORDER OF
    MAGNITUDE: scaled NREL equipment list; published plants differ by 2-5x) and electricity. configuration
    'anaerobic_aerobic' | 'aerobic'; cod_removal = the user's fractions of the COD entering each stage {anaerobic,
    anaerobic_to_biomass, aerobic, aerobic_to_biomass} (published values the caller may pass explicitly: 0.91, 0.05,
    0.96, 0.22 from Humbird 2011). Optional water-reuse scope (reverse osmosis, brine evaporator), aerator efficiency,
    electricity price + hours and biogas value for the yearly cost. Returns stages, biogas, capital.items for
    capital_investment (already installed, F_BM 1), power, operating, evidence and warnings."""
    return json.dumps(TUC.wastewater_treatment_cost(flow_m3_s, cod_kg_m3, configuration, cod_removal, include_reverse_osmosis,
                                                    include_brine_evaporator, density_kg_m3, electricity_price_usd_kWh,
                                                    operating_hours_per_year, aeration_kg_O2_per_kWh, biogas_value_usd_GJ,
                                                    year, cepci), default=float, ensure_ascii=False)


AGENT_WRAPPERS = [cooling_tower_design, steam_boiler_design, wastewater_treatment_cost]
