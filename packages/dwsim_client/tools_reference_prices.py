"""Agent tools for reference prices and design basis (bloque M26_P). No DWSIM flowsheet needed.
Engine: `packages.costing.reference_prices` (bank of published, dated values in data/reference_prices.json).

`reference_prices` proposes published prices of raw materials, products and services; `design_basis_defaults` proposes the
fiscal, financial, labour and design-basis values. Both return every value FLAGGED as an assumption the user overrides."""
from __future__ import annotations

from datetime import date

from packages.costing import reference_prices as RP


def _today(today: str | None) -> date | None:
    if today is None:
        return None
    try:
        return date.fromisoformat(today)
    except ValueError as exc:
        raise ValueError(f"today = {today!r}: usa formato ISO AAAA-MM-DD.") from exc


def reference_prices(items: list[str], region: str = "MX", currency: str = "USD", exchange_rates: dict | None = None,
                     today: str | None = None) -> dict:
    """Precios de REFERENCIA publicados (no cotizaciones) para proponer un primer cuadro económico cuando el usuario no ha dado
    precios. Cada valor trae fuente, URL, periodo, fecha de lectura, antigüedad y la marca «VALOR DE REFERENCIA PUBLICADO, NO TU
    COTIZACIÓN»: preséntalo siempre como suposición que el usuario puede cambiar. Lo que no está en el banco NO se inventa.

    items: nombres en español o inglés, por ejemplo ['castor oil', 'aceite de soya', 'used cooking oil', 'methanol', 'ethanol',
      'biodiesel', 'glycerol', 'naoh', 'sulfuric acid', 'hexane', 'sugarcane', 'raw sugar', 'corn', 'sorghum', 'ddgs',
      'process water', 'wastewater', 'diesel', 'minimum wage', 'average wage', 'natural gas', 'electricity', 'fix'].
    region: 'MX-SIN' (Sinaloa), 'MX', 'US' o 'WORLD'. Si no hay valor para la región se usa el siguiente en el orden
      MX-SIN → MX → US → WORLD y se dice (fallback_region_used, fallback_note).
    currency: moneda en que quieres value_converted ('USD' o 'MXN'). La conversión MXN↔USD usa el FIX de Banxico fechado del
      banco, salvo que des exchange_rates = {'MXN': <currency por 1 MXN>} (o {'USD': ...}); siempre dice cuál usó.
    today: fecha ISO para calcular la antigüedad (por defecto, hoy).

    Devuelve values {item: {found, value, unit, currency, basis, period {start, end}, source, url, date_read, source_tier
    ('primaria'|'secundaria'), low/high si hay rango, published_value/published_unit/conversion, value_converted,
    unit_converted, exchange_rate_applied, age_days, stale, region_used, fallback_region_used, flag, warnings, notes}}, found,
    missing [{item, reason, close_matches}], assumptions (frases listas para mostrar), warnings, not_included, method y
    evidence. Unidades: sólidos y aceites en USD/t o MXN/t; líquidos publicados por galón o litro quedan por volumen (USD/m3,
    MXN/m3: no se supone densidad); agua en MXN/m3; salarios en MXN/day. 'sugarcane' en México viene por tonelada de azúcar
    recuperable (multiplica por el KARBE de tu ingenio ÷ 1000). 'natural gas' y 'electricity' se leen del banco de servicios
    (utility_costs / list_utility_price_references); para México la electricidad es una tarifa horaria y se remite a él."""
    return RP.suggest(items, region, currency, exchange_rates, _today(today))


def design_basis_defaults(region: str = "MX", plant_type: str | None = None, today: str | None = None) -> dict:
    """Base de diseño y parámetros fiscales, financieros y laborales PUBLICADOS para arrancar la evaluación económica de una
    planta sin preguntar cada número. Todo se devuelve marcado como suposición con su fuente y URL.

    region: 'MX-SIN', 'MX', 'US' o 'WORLD' (para 'US' solo hay las convenciones de NREL; lo fiscal y laboral es de México).
    plant_type: texto libre ('biodiesel', 'bioetanol de caña', 'ingenio', 'aceite', 'alimentos'...) para PROPONER la fracción de
      depreciación fiscal de maquinaria (LISR arts. 34-35) por palabras clave; es una propuesta, no un criterio fiscal.
    today: fecha ISO para la antigüedad.

    Devuelve values {id: valor con fuente} con, entre otros: isr_corporate_mx (0.30), ptu_mx (0.10), depreciation_*_mx
    (fracción máxima anual), banxico_target_rate, tiie_28, cetes_28, fx_fix_mxn_usd (MXN/USD), inflation_mx,
    equity_risk_premium_mx, discount_rate_nrel, on_stream_hours (h/y), plant_life (y), construction_schedule (años y fracciones
    por año en table), working_capital_fraction, startup_time, equity_fraction, zafra_duration_mx (d/y), location_factor_mx,
    cepci_2024_06, minimum_wage_mx e imss_average_wage_mx (MXN/day). Además by_kind, suggested_depreciation, not_found (lo
    buscado y no hallado, por ejemplo un WACC publicado para México), assumptions, warnings, not_included, method y evidence."""
    return RP.design_basis(region, plant_type, _today(today))


REFERENCE_PRICES_TOOLS = [reference_prices, design_basis_defaults]
