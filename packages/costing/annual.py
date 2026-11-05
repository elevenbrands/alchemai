"""Capital and annual cost of a flowsheet: purchase cost → fixed capital (Lang factor × location factor) →
annualized capital (capital recovery factor) + operating cost (electricity, heating, cooling) = total annual cost.

Lang factors (fixed capital or total capital from delivered equipment cost), only the values checked against a source:
  lang_1948_fluid 4.74 and lang_1948_solid 3.10 (Lang's original fixed-capital factors), peters_2003_fluid_fci 5.0 and
  peters_2003_fluid_tci 6.0 (Peters, Timmerhaus & West revision, fluid plants). Any other value: pass `lang_factor`.
Location factors: IDAES's table (Seider Table 16.13 for U.S. regions + international sources), e.g. Mexico 0.84-0.92.
Prices and hours are inputs, never defaults: the user's prices decide the answer.
"""
from __future__ import annotations

LANG = {"lang_1948_fluid": 4.74, "lang_1948_solid": 3.10, "peters_2003_fluid_fci": 5.0, "peters_2003_fluid_tci": 6.0}


def location_factor(country: str, city: str = "aggregate", which: str = "mid") -> float:
    from idaes.core.base.costing_base import load_location_factors
    lf = load_location_factors()
    if country not in lf:
        raise ValueError(f"country {country!r} not in IDAES location factors")
    if city not in lf[country]:
        raise ValueError(f"{country}: cities {sorted(lf[country])}")
    v = lf[country][city]
    if which == "mid" and "min" in v and "max" in v:
        return 0.5 * (v["min"] + v["max"])
    return float(v[which]) if which in v else float(next(iter(v.values())))


def crf(interest: float, years: int) -> float:
    """Capital recovery factor i(1+i)^n / ((1+i)^n − 1)."""
    if interest == 0:
        return 1.0 / years
    g = (1 + interest) ** years
    return interest * g / (g - 1)


def annual_cost(purchase_costs_usd: dict[str, float], energy_kW: dict[str, tuple[str, float]], prices: dict[str, float],
                hours_per_year: float, interest: float, years: int, lang: str | float = "lang_1948_fluid",
                location: float = 1.0) -> dict:
    """energy_kW: unit → (kind, kW) with kind ∈ electricity | heating | cooling. prices: electricity_usd_kWh,
    heating_usd_GJ, cooling_usd_GJ (only those used are required)."""
    f_lang = float(lang) if not isinstance(lang, str) else LANG[lang]
    purchase = sum(purchase_costs_usd.values())
    fci = purchase * f_lang * location
    opex = {}
    for unit, (kind, kw) in energy_kW.items():
        kw = abs(kw)
        if kind == "electricity":
            p = prices["electricity_usd_kWh"]
            opex[unit] = kw * hours_per_year * p
        elif kind in ("heating", "cooling"):
            p = prices[f"{kind}_usd_GJ"]
            opex[unit] = kw * hours_per_year * 3600.0 / 1e6 * p
        else:
            raise ValueError(f"{unit}: energy kind {kind!r}")
    f = crf(interest, years)
    ann_cap = fci * f
    tot_opex = sum(opex.values())
    return {"purchase_cost_usd": purchase, "lang_factor": f_lang, "location_factor": location, "fixed_capital_usd": fci,
            "capital_recovery_factor": f, "annualized_capital_usd_y": ann_cap, "operating_cost_usd_y": tot_opex,
            "operating_by_unit_usd_y": opex, "total_annual_cost_usd_y": ann_cap + tot_opex,
            "purchase_by_unit_usd": purchase_costs_usd}
