"""Agent tool for the one-question dossier (bloque M27): see `packages.dossier.dossier`."""
from __future__ import annotations

from packages.dossier import dossier as DS


def process_dossier(subject: str, refinery: str | None = None, plant: dict | None = None, region: str = "MX",
                    prices: dict | None = None, plant_data: list | None = None, period: str | None = None,
                    out_dir: str | None = None, theme: str = "light",
                    power_plant: str | None = None, year: int | None = None) -> dict:
    """TODO de una sola pregunta («el proceso completo de la refinería de Tula, su rentabilidad y oportunidades»): corre el
    modelo (refinery='tula' u otra de list_refineries, en modo «como opera»; o plant={'model': 'biodiesel'|
    'sugarcane_ethanol'|'corn_ethanol'|'cornstover_ethanol', …argumentos de ese modelo}; o power_plant='laguna_verde' con year), su economía regional, el escáner
    de oportunidades (más, en refinerías, el valor de la capacidad ociosa de cada unidad calibrada), y dibuja el diagrama
    de flujo (vista por secciones y completa) con las oportunidades numeradas sobre su equipo. Escribe <out_dir>/<tema>.md
    y dos .svg (por omisión ~/Documents/AlchemAI/dossiers) y devuelve summary (márgenes, validación contra lo publicado,
    parámetros calibrados, CO2), opportunities, rutas de los archivos y evidence. plant_data: lecturas de planta
    [{unit_id, variable, value, uom, period}] para evaluar integración de calor, hornos, vapor y energía por unidad.
    No agrega física: encadena motores validados por separado y conserva sus banderas."""
    return DS.process_dossier(subject, refinery, plant, region, prices, plant_data, period, out_dir, theme, power_plant, year)


DOSSIER_TOOLS = [process_dossier]
