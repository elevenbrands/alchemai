"""Agent tool for the design-heuristics review (`packages.review.heuristics`)."""
from __future__ import annotations

from packages.review import heuristics as H


def design_heuristics_review(checks: list[dict] | None = None, flowsheet_id: str | None = None,
                             list_areas: list[str] | None = None) -> dict:
    """Revisión del diseño con HEURÍSTICAS de ingeniería (criterio de ingeniero): compara valores del diseño con la
    práctica usual publicada y dice cuáles quedan fuera, con la fuente de cada regla. No calcula el diseño: lo critica.
    Úsala al terminar un diseño o un flowsheet, antes de reportarlo, y cuando el usuario pregunte si algo es razonable.

    checks: [{rule_id, value, label?}] con valores en las unidades de la regla (R/R_min, N/N_min, NPSHA/NPSHR,
      tiempo de residencia, velocidades, U, ΔP...). flowsheet_id: flowsheet de DWSIM resuelto; se extraen solos la
      relación de compresión y la T de descarga de compresores, el acercamiento mínimo de intercambiadores, la T de
      salida de enfriadores y calentadores, las T de condensador y rehervidor de columnas y, en columnas de atajo, R/R_min
      y N/N_min. list_areas: [] para ver todo el catálogo (o ["distillation", "heat_exchangers", ...]) con rule_id, intervalo y
      unidades, para escoger qué revisar.

    Devuelve findings ordenados (warning = fuera de la práctica usual, info = difiere del valor típico, ok), cada uno con
    la regla, el valor, el intervalo, el mensaje y la fuente (documento consultado con URL y la obra que cita), y notes.
    Un warning NO es un error: pide justificar o revisar esa decisión. Reporta al usuario los warnings con su fuente."""
    try:
        return H.review(checks, flowsheet_id, list_areas)
    except (ValueError, KeyError) as ex:
        return {"ok": False, "error": str(ex)}


REVIEW_TOOLS = [design_heuristics_review]
