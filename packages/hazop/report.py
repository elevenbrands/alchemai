"""HAZOP table rendering (Markdown, Excel) and the `hazop_preliminary` tool."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.hazop.graph import build_graph  # noqa: E402
from packages.hazop.rules import HazopRow, hazop_rows  # noqa: E402

COLUMNS = ["Nodo", "Tipo", "Parámetro", "Desviación", "Causas", "Consecuencias", "Salvaguardas", "Sev.", "Evidencia", "Fuente"]


def to_markdown(rows: list[HazopRow], title: str = "HAZOP preliminar") -> str:
    lines = [f"# {title}", "", "Generado por reglas deterministas sobre el flowsheet resuelto. **No sustituye un HAZOP con equipo multidisciplinario.** "
             "Las filas `rule` salen de reglas por tipo de equipo con los números del caso; cualquier fila `proposal` es una sugerencia del LLM sin verificar.", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join([r.node, r.kind, r.parameter, r.deviation, "<br>".join(r.causes), "<br>".join(r.consequences),
                                         "<br>".join(r.safeguards), str(r.severity), r.evidence, r.source]) + " |")
    return "\n".join(lines) + "\n"


def to_excel(rows: list[HazopRow], path: str | Path) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "HAZOP"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E79")
    for r in rows:
        ws.append([r.node, r.kind, r.parameter, r.deviation, "\n".join(r.causes), "\n".join(r.consequences), "\n".join(r.safeguards), r.severity, r.evidence, r.source])
    widths = [14, 14, 12, 26, 40, 48, 40, 6, 36, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(str(path))
    return str(path)


def hazop_from_flowsheet(fs, name: str = "flowsheet") -> tuple[list[HazopRow], dict]:
    g = build_graph(fs, name)
    return hazop_rows(g), g.to_dict()


# ------------------------------------------------------------------ tool (agent / MCP)
def hazop_preliminary(flowsheet_id: str, excel_path: str | None = None) -> dict:
    """Preliminary HAZOP of a SOLVED flowsheet: deterministic deviation rules (CCPS guide words × parameters)
    per unit, instantiated with the solved T, P, phases, duties and hazard flags (flammable, toxic, hydrates).
    Returns the rows (node, parameter, deviation, causes, consequences, safeguards, severity 1-5, evidence) and
    a Markdown table; optionally writes an .xlsx. Severity is a preliminary ranking, not a risk assessment.
    Call solve() first."""
    from packages.dwsim_client import tools as T
    s = T._s(flowsheet_id)
    rows, graph = hazop_from_flowsheet(s["fs"], flowsheet_id)
    out = {"n_rows": len(rows), "rows": [r.to_dict() for r in rows], "markdown": to_markdown(rows),
           "graph_units": {k: {"kind": v["kind"], "inlets": v["inlets"], "outlets": v["outlets"], "duty_kW": v["duty_kW"]} for k, v in graph["units"].items()}}
    if excel_path:
        out["excel"] = to_excel(rows, excel_path)
    return out
