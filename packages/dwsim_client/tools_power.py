"""Agent tools for power-plant models from public data (bloque M27_N). No DWSIM flowsheet needed.
Engine: `packages.power.nuclear` (BWR direct steam cycle with CoolProp IAPWS-95, Laguna Verde from IAEA PRIS / CNSNS / CFE /
CENACE / NEI / SEMARNAT public data). `power_plant_model` returns the ProcessModel JSON that the diagram (M27_D) draws."""
from __future__ import annotations

from packages.power import nuclear as NU


def power_plant_model(plant: str = "laguna_verde", year: int | None = None, unit: str | int | None = None,
                      overrides: dict | None = None, prices: dict | None = None, today: str | None = None) -> dict:
    """Modelo de una central eléctrica con datos PÚBLICOS: qué hace, su ciclo de vapor, su generación, su rentabilidad de
    operación, el CO2 que evita, su diagrama de proceso (process_model) y oportunidades de mejora. Hoy: 'laguna_verde'
    (Central Nucleoeléctrica Laguna Verde, CFE, Veracruz: 2 reactores BWR-5 de 2317 MWt).

    year: año de operación publicado por IAEA PRIS (1990-2025; por defecto el último). unit: '1', '2' o None (ambas).
    overrides (todas opcionales; sin ellas se usan los datos públicos y las simplificaciones declaradas):
      n_feedwater_heaters (2-9, por defecto 6), crossover_pressure_Pa, throttle_pressure_Pa, reheat_ttd_K, heater_ttd_K,
      condenser_ttd_K (K, por defecto 0 = límite termodinámico), seawater_inlet_K (por defecto 299.65 K, promedio de las
      temporadas publicadas), cooling_flow_m3_s_unit (por defecto 31.5), generator_efficiency (fracción; por defecto
      incluida en la eficiencia calibrada), pump_efficiency, thermal_MWt, gross_MWe_target (MWe a los que se calibra la
      turbina; por defecto la bruta de PRIS), heaters_out (lista de calentadores fuera de servicio, 1 = menor presión).
    prices: sustituye sugerencias: 'energia_usd_mwh' o 'energia_mxn_mwh' (por defecto PND promedio de la zona Veracruz,
      CENACE, del año si está cargado: 2025, 2026), 'fx_mxn_usd', 'combustible_usd_mwh', 'operacion_usd_mwh',
      'capital_usd_mwh' (por defecto NEI/EUCG 2025, promedio BWR de EE. UU.). today: fecha ISO para el tipo de cambio.

    Devuelve plant (unidades, potencias, licencias, enfriamiento), process_model (ProcessModel para process_diagram),
    cycle (tabla de estados P/T/h/s/x, flujo de vapor, eficiencia bruta, heat rate kJ/kWh y Btu/kWh, presión y carga del
    condensador, agua de mar, sensibilidades a número de calentadores, a la temperatura del mar y a un calentador fuera),
    generation (por unidad: GWh publicados vs neta × horas × factor de carga), economics (ingreso, combustible, operación,
    capital, márgenes USD/MWh y USD/año, recuperación del aumento de potencia, precios con fuente/URL/periodo marcados como
    sugerencia), emissions_avoided (t CO2/año con el factor oficial SEMARNAT/CRE), opportunities (contrato de
    oportunidades, con unit_ids del process_model), calibrated_parameters (eficiencia isentrópica de turbina CALIBRADA a la
    bruta publicada), published_comparison (validación contra cifras no usadas en el ajuste), not_found, assumptions,
    warnings, not_included, method y evidence."""
    return NU.run_plant(plant=plant, year=year, unit=unit, overrides=overrides, prices=prices, today=today)


def list_power_plants() -> dict:
    """Qué centrales eléctricas tienen datos públicos cargados para power_plant_model, con sus unidades (tipo, modelo,
    MWt, MWe bruta y neta, licencia), años de operación de PRIS disponibles, fuentes con URL y datos que no se encontraron."""
    d = NU._load()
    years = {u: [min(v), max(v)] for u, v in d["pris_annual"].items() if u in d["units"]}
    return {"plants": {d["plant"]: {"name": d["name"], "units": d["units"], "pris_years": years,
                                    "market_price_years": sorted(d["market_price"]["by_year"]), "model_available": True}},
            "sources": d["sources"], "not_found": d["not_found"], "date_read": d["date_read"], "warnings": [],
            "not_included": ["Otras centrales de CFE o privadas: no cargadas."],
            "method": "Lectura del archivo de datos públicos packages/power/data/laguna_verde.json (sin cálculo).",
            "evidence": {"verified": True, "reasons": [], "n_plants": 1,
                         "note": "verified = los datos cargados existen y son legibles; no certifica la fuente."}}


POWER_TOOLS = [power_plant_model, list_power_plants]
