"""One command from the plant's workbook to the report (bloque M21).

    uv run python -m packages.energy.steam_cli plantilla datos_planta.xlsx
    uv run python -m packages.energy.steam_cli diagnostico datos_planta.xlsx --salida carpeta/
    uv run python -m packages.energy.steam_cli ejemplo carpeta/

`diagnostico` writes reporte.html (self-contained), reporte.pdf (if a Chrome is installed), reporte.md and resultado.json.
Missing data never crash it: the engine either refuses with the list of what is missing or reports what each gap prevents.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from packages.energy import steam_diagnostic as SD
from packages.energy import steam_html as SH
from packages.energy import steam_report as SR
from packages.energy import steam_workbook as SW

ILLUSTRATIVE_NOTICE = ("CASO ILUSTRATIVO: los datos de esta planta son INVENTADOS para mostrar el formato. No es un cliente ni "
                       "una validación.")
# An invented plant in plant units (MOCKS ENTRADAS-ILUSTRATIVAS-M21). {field: (value, unit)}; tables under __leaks__/__pipes__
EXAMPLE = {
    "plant_label": ("Planta de ejemplo (datos inventados)", None), "currency": ("MXN", None), "altitude": (540, "m"),
    "fuel_name": ("Gas natural", None), "fuel_T": (25, "°C"),
    "fuel_bill_amount": (1_425_000, None), "fuel_bill_period": ("octubre 2026", None), "fuel_bill_energy": (9_500, "GJ"),
    "fuel_bill_basis": ("Superior (PCS)", None), "fuel_bill_volume": (248_000, "m³"), "fuel_bill_vol_T": (20, "°C"),
    "fuel_bill_vol_P": (1.0, "kg/cm² abs."), "water_price": (45, None), "elec_price": (2.4, None),
    "aux_power": (6, "kWh por tonelada de vapor"),
    "steam_pressure": (8, "kg/cm² man."), "superheat": (0, "°C"), "blowdown": (8, "%"),
    "rated_capacity": (600, "CC (caballos caldera)"),
    "o2": (7, "%"), "o2_basis": ("Seca", None), "stack_T": (220, "°C"), "air_T": (25, "°C"), "air_RH": (50, "%"),
    "setting_loss_unknown": (SW.YES, None),
    "fw_mode": ("Desaireador o tanque calentado con vapor de la caldera", None), "fw_T": (105, "°C"), "makeup_T": (20, "°C"),
    "cond_return": (40, "%"), "cond_T": (80, "°C"), "hours": (6000, "h al año"), "steam_production": (6, "t/h"),
    "m1_published": (SW.YES, None), "m2_published": (SW.YES, None), "m3_blowdown_target": (5, "%"),
    "m4_flash_pressure": (0.3, "kg/cm² man."), "m4_drain_T": (40, "°C"), "m5_return_target": (70, "%"),
    "ambient_T": (28, "°C"), "wind": (0, "m/s"),
    "__leaks__": [["Fuga de vapor", "cabezal principal", 4, 3, "mm", 8, "kg/cm² man."],
                  ["Trampa de vapor fallada (abierta)", "línea de proceso 2", 3, 4, "mm", 8, "kg/cm² man."]],
    "__pipes__": [["línea a marmitas", 114.3, "mm", 30, "m", 8, "kg/cm² man.", 50, "mm", 0.045],
                  ["bajante a lavado", 60.3, "mm", 12, "m", 8, "kg/cm² man."]],
}


def run_workbook(workbook: str, out_dir: str, notice: str | None = None, pdf: bool = True, today: str | None = None) -> dict:
    """Workbook -> files in out_dir. Returns their paths and the headline numbers. Raises ValueError in Spanish."""
    read = SW.read_workbook(workbook)
    inputs = dict(read["inputs"])
    if today:
        inputs["today"] = today
    if read["missing_essentials"]:
        raise ValueError("A la hoja le faltan datos indispensables (sin ellos no hay diagnóstico; no se suponen):\n- "
                         + "\n- ".join(read["missing_essentials"]))
    report = SR.steam_diagnostic_report(inputs=inputs, plant_label=read["plant_label"], notice=notice)
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, name) for k, name in (("html", "reporte.html"), ("markdown", "reporte.md"),
                                                             ("json", "resultado.json"), ("pdf", "reporte.pdf"))}
    with open(paths["markdown"], "w", encoding="utf-8") as fh:
        fh.write(report["markdown"])
    with open(paths["html"], "w", encoding="utf-8") as fh:
        fh.write(SH.steam_report_html(report, read["plant_label"], read["notes"]))
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"intake_notes": read["notes"], "diagnostic": report["diagnostic"]}, fh, ensure_ascii=False, indent=1,
                  default=float)
    pdf_result = SH.html_to_pdf(paths["html"], paths["pdf"]) if pdf else {"ok": False, "reason": "no se pidió"}
    if not pdf_result["ok"]:
        paths.pop("pdf")
    d = report["diagnostic"]
    return {"files": paths, "pdf": pdf_result, "headline": report["headline"], "currency": d["currency"],
            "verified": report["verified"], "assumed_values": [v["item"] for v in d["assumed_values"]],
            "data_flags": [f["message"] for f in d["data_review"]["flags"]],
            "not_evaluated": [m["label"] for m in d["measures"] if m["status"] != "evaluated"],
            "missing_data": [m["item"] for m in d["missing_data"]], "intake_notes": read["notes"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alchemai-vapor", description="Diagnóstico de vapor: de la hoja de Excel al reporte.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plantilla", help="escribe la hoja de Excel vacía que llena la planta")
    p.add_argument("archivo")
    p = sub.add_parser("diagnostico", help="lee la hoja llena y escribe el reporte")
    p.add_argument("archivo")
    p.add_argument("--salida", default=None, help="carpeta de salida (por defecto, junto al archivo)")
    p.add_argument("--sin-pdf", action="store_true")
    p = sub.add_parser("ejemplo", help="escribe una hoja llena con una planta INVENTADA y su reporte")
    p.add_argument("carpeta")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "plantilla":
            print("Plantilla escrita en", SW.build_template(a.archivo))
            return 0
        if a.cmd == "ejemplo":
            os.makedirs(a.carpeta, exist_ok=True)
            book = SW.build_template(os.path.join(a.carpeta, "datos_planta_ejemplo.xlsx"), EXAMPLE)
            out = run_workbook(book, a.carpeta, notice=ILLUSTRATIVE_NOTICE)
        else:
            out = run_workbook(a.archivo, a.salida or os.path.join(os.path.dirname(os.path.abspath(a.archivo)), "reporte_vapor"),
                               pdf=not a.sin_pdf)
    except ValueError as e:
        print("No se pudo calcular:\n" + str(e), file=sys.stderr)
        return 2
    h, cur = out["headline"], out["currency"]
    print("Reporte listo:")
    for kind, path in out["files"].items():
        print(f"  {kind}: {path}")
    if not out["pdf"]["ok"]:
        print("  PDF no generado:", out["pdf"]["reason"])
    print(f"Costo del vapor: {h['cost_per_tonne']:,.1f} {cur} por tonelada; eficiencia {100 * h['efficiency']:.1f} %."
          + (f" Costo anual {h['annual_cost']:,.0f} {cur}." if "annual_cost" in h else ""))
    for label, items in (("Valores supuestos", out["assumed_values"]), ("Datos por confirmar", out["data_flags"]),
                         ("Medidas no evaluadas", out["not_evaluated"])):
        if items:
            print(f"{label}: " + "; ".join(items))
    print("Balances comprobados:", "sí" if out["verified"] else "NO, revisar el reporte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
