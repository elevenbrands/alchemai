"""Agent tools for whole plants (bloque M26_B): published reference biorefineries and a vegetable-oil biodiesel plant.
No DWSIM flowsheet needed. Engines: `packages.plants.reference` and `packages.plants.biodiesel` (BioSTEAM and its
`biorefineries` package, imported lazily; BioSTEAM's global state is restored after every call)."""
from __future__ import annotations


def list_reference_plants() -> dict:
    """Catálogo de plantas completas disponibles, sin simular nada: para cada modelo publicado (`sugarcane_ethanol`,
    `corn_ethanol`, `cornstover_ethanol`) su descripción, la cita del modelo, la fuente de los valores publicados con los
    que se compara y esos valores; además los parámetros económicos que acepta `tea=`, las claves de `prices=` y los
    aceites que acepta `biodiesel_plant` (`oils`). Úsela antes de `reference_biorefinery` si no sabe qué planta pedir."""
    from packages.plants import biodiesel as BD
    from packages.plants import reference as RF
    out = RF.list_reference_plants()
    out["oils"] = BD.list_oils()
    return out


def reference_biorefinery(plant: str, feed_flow_kg_h: float | None = None, operating_hours_y: float | None = None,
                          prices: dict | None = None, tea: dict | None = None, year: int | None = None,
                          cepci: float | None = None) -> dict:
    """Planta completa PUBLICADA, resuelta con BioSTEAM: `plant` = sugarcane_ethanol (etanol de caña con cogeneración,
    Huang 2016), corn_ethanol (molienda seca de maíz, Kwiatkowski 2006) o cornstover_ethanol (etanol celulósico de rastrojo,
    Humbird/NREL 2011). Sin más argumentos da la línea base publicada. Opcionales, todos sustituyen un valor del modelo:
    feed_flow_kg_h (kg/h de la materia prima tal como entra, húmeda; reescala y recostea toda la planta),
    operating_hours_y (h/año), prices {ID de corriente: USD/kg, 'electricity': USD/kWh, 'steam' y 'natural_gas': USD/kg}
    (los ID válidos son los de `streams`, p. ej. sugarcane, corn, cornstover, ethanol, DDGS), tea {discount_rate,
    income_tax, plant_life_years, depreciation ('MACRS7', 'SL10'…), lang_factor, construction_schedule,
    working_capital_over_fci, startup_months, finance_fraction, finance_interest, finance_years, labor_cost_usd_y,
    maintenance, property_tax, property_insurance, administration}; year (2015-2023, tabla CEPCI de IDAES) o cepci (valor
    del índice) llevan el costo de equipo a ese año; si ambos son None el capital queda en el año del modelo publicado.
    Devuelve design_basis (cada valor con su origen: modelo publicado o usuario), streams (alimentaciones, productos y
    corrientes entre secciones: kg/h, K, Pa, fracciones másicas), sections, products, utilities (potencia consumida,
    generada y exportada en W; vapor y enfriamiento por agente), equipment (lista que acepta capital_investment: name,
    F_BM de BioSTEAM, purchase_cost_usd, base_purchase_cost_usd, installed_cost_usd, cepci), capital (compra, DPI, TDC, FCI,
    TCI), operating_cost, production_cost, economics (VPN e TIR a los precios dados, precio mínimo de venta en USD/kg, /L y
    /gal), published_benchmark (desviación de la línea base frente a lo publicado), evidence (cierre de masa, primera
    ley, VPN = 0 al precio mínimo), warnings, assumptions, not_included. Precios y parámetros del modelo son SUGERENCIAS
    con fuente: dígale al usuario cuáles se usaron. La primera llamada tarda ~20-60 s (carga de BioSTEAM)."""
    from packages.plants import reference as RF
    return RF.reference_biorefinery(plant, feed_flow_kg_h, operating_hours_y, prices, tea, year, cepci)


def biodiesel_plant(oil_feed_kg_h: float | None = None, oil: str | None = None, ffa_mass_fraction: float | None = None,
                    water_mass_fraction: float | None = None, operating_hours_y: float | None = None,
                    prices: dict | None = None, tea: dict | None = None, conversion_per_stage: float | None = None,
                    methanol_excess: float | None = None, catalyst_mass_fraction_of_oil: float | None = None,
                    oil_temperature_K: float = 298.15, year: int | None = None, cepci: float | None = None) -> dict:
    """Planta de biodiésel (FAME) + glicerol crudo al 80 % por transesterificación alcalina con metanol y metóxido de sodio
    (sección publicada de BioSTEAM: dos reactores a 60 °C con centrífuga intermedia, lavado, secado al vacío, recuperación
    de metanol). OBLIGATORIOS (datos del usuario, no se inventan): oil_feed_kg_h (kg/h de aceite tal como llega), oil
    ('triolein' = aceite oleico genérico; 'tripalmitin' = grasa saturada, caso límite; 'castor' | 'ricino' | 'higuerilla'
    = aceite de ricino como PSEUDO-TRIGLICÉRIDO DECLARADO: masa molar de la trirricinoleína y demás propiedades prestadas de
    la trioleína, así se informa en assumptions), ffa_mass_fraction (ácidos grasos libres, fracción másica; por encima de
    0.01 la herramienta se niega: falta el pretratamiento ácido), water_mass_fraction, operating_hours_y. Opcionales:
    prices {oil, biodiesel, crude_glycerol, methanol, catalyst, HCl, NaOH, wash_water, wastewater: USD/kg; electricity:
    USD/kWh; steam: USD/kg}, tea (mismas claves que reference_biorefinery), conversion_per_stage (0-1, sugerido 0.90),
    methanol_excess (sugerido 1.0 = 6:1 molar), catalyst_mass_fraction_of_oil (kg NaOCH3/kg triglicérido; sugerido 0.25 %
    + 0.19·%AGL), oil_temperature_K, year | cepci (año del costo de equipo; None = CEPCI 567.5). El precio del aceite NO se
    sugiere: sin prices['oil'] se devuelve solo el margen de transformación (precio mínimo sin aceite) y VPN/TIR en None.
    Los demás precios y parámetros económicos se SUGIEREN con fuente (Huang et al. 2016, precios de 2013; NREL) y vienen
    marcados en design_basis. Devuelve lo mismo que reference_biorefinery más oil_model y yields (kg de éster, glicerol,
    metanol, catalizador por kg de aceite; conversión; densidad; L/año). Para ricino lea warnings y not_included: la
    viscosidad y la separación del glicerol reales NO están modeladas."""
    from packages.plants import biodiesel as BD
    return BD.biodiesel_plant(oil_feed_kg_h, oil, ffa_mass_fraction, water_mass_fraction, operating_hours_y, prices, tea,
                              conversion_per_stage, methanol_excess, catalyst_mass_fraction_of_oil, oil_temperature_K, year,
                              cepci)


PLANTS_TOOLS = [reference_biorefinery, biodiesel_plant, list_reference_plants]
