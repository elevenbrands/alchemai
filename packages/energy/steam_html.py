"""The steam diagnostic as one self-contained HTML page, printable to PDF (bloque M21).

The page is the Markdown report of `steam_report` (every number comes from there, unchanged) plus a row of headline tiles
and bar charts drawn as inline SVG from the same diagnostic dict. No number is computed here. No external file is loaded:
the HTML can be e-mailed or opened offline. `html_to_pdf` prints it with a locally installed Chrome; without one, the
HTML is still the deliverable and the function says so.
"""
from __future__ import annotations

import html
import os
import re
import shutil
import subprocess

from packages.energy import steam_report as SR

CHROME_PATHS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--series:#2a78d6;
--note:#fff6dc;--note-edge:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--grid:#2c2c2a;--series:#3987e5;--note:#2c2717;--note-edge:#c98500}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
header.top{border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:18px}
header.top .brand{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}
h1{font-size:26px;margin:4px 0 0}h2{font-size:19px;margin:30px 0 8px;padding-top:8px;border-top:1px solid var(--grid)}
p{margin:8px 0}ul{margin:8px 0;padding-left:20px}li{margin:3px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0 6px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:10px 12px}
.tile .v{font-size:22px;font-weight:650;line-height:1.2}.tile .l{font-size:12px;color:var(--ink2)}
.tile .s{font-size:12px;color:var(--muted)}
blockquote{margin:12px 0;padding:10px 14px;background:var(--note);border-left:4px solid var(--note-edge);border-radius:4px}
blockquote p{margin:0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px;background:var(--surface)}
th,td{border:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
figure{margin:14px 0;background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:12px 12px 6px}
figcaption{font-weight:600;margin-bottom:2px}figure .sub{font-size:12.5px;color:var(--ink2);margin-bottom:6px}
svg{display:block;width:100%;height:auto}svg text{font:12.5px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.bar{fill:var(--series)}.cat{fill:var(--ink2)}.val{fill:var(--ink)}.axis{stroke:var(--grid);stroke-width:1}
footer{margin-top:28px;font-size:12px;color:var(--muted);border-top:1px solid var(--grid);padding-top:8px}
@media print{:root{--surface:#fff;--page:#fff;--ink:#0b0b0b;--ink2:#52514e;--grid:#d5d4cd;--series:#2a78d6;--note:#fff6dc}
body{font-size:11.5px}main{max-width:none;padding:0}h2{break-after:avoid}tr,figure,blockquote,.tile{break-inside:avoid}
table{font-size:10.5px;width:99.6%}.scroll{overflow:visible}@page{size:Letter;margin:14mm}}
"""


# ------------------------------------------------------------------------------------------------ markdown subset
def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


def markdown_to_html(md: str) -> list[tuple[str, str]]:
    """The Markdown subset `steam_report` writes -> [(heading text or '', html of that block)], split at level-2 headings."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            i += 1                                                               # the page header carries the title
            continue
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
            i += 1
            continue
        buf = sections[-1][1]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            buf.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            buf.append("<ul>" + "".join(items) + "</ul>")
            continue
        if line.startswith("> "):
            buf.append(f"<blockquote><p>{_inline(line[2:])}</p></blockquote>")
        elif line.strip():
            buf.append(f"<p>{_inline(line)}</p>")
        i += 1
    return [(title, "\n".join(parts)) for title, parts in sections]


# --------------------------------------------------------------------------------------------------------- charts
def bar_chart(rows: list[tuple[str, float, str]], title: str, subtitle: str) -> str:
    """Horizontal bars, one series, sorted as given. rows: (category, value, value label). Values must be >= 0."""
    rows = [r for r in rows if r[1] is not None and r[1] > 0]
    if not rows:
        return ""
    label_w, value_w, width, bar_h, gap = 400, 130, 960, 16, 12
    plot_w = width - label_w - value_w
    top = max(v for _, v, _ in rows)
    height = len(rows) * (bar_h + gap) + gap
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" y2="{height}"/>']
    for n, (cat, value, label) in enumerate(rows):
        y = gap + n * (bar_h + gap)
        w = max(2.0, plot_w * value / top)
        r = min(4.0, w / 2)
        tip = html.escape(f"{cat}: {label}")
        parts.append(f'<text class="cat" x="{label_w - 8}" y="{y + bar_h - 3.5}" text-anchor="end">{html.escape(_clip(cat, 62))}</text>')
        parts.append(f'<path class="bar" d="M{label_w},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{bar_h - 2 * r:.1f} '
                     f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"><title>{tip}</title></path>')
        parts.append(f'<text class="val" x="{label_w + w + 8:.1f}" y="{y + bar_h - 3.5}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return (f"<figure><figcaption>{html.escape(title)}</figcaption><div class=\"sub\">{html.escape(subtitle)}</div>"
            + "".join(parts) + "</figure>")


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _charts(d: dict) -> dict:
    """Charts keyed by the report section they belong to. Every value is read from the diagnostic dict."""
    cur, boiler = d["currency"], d["baseline"]["boiler"]
    basis = SR.BASIS_ES[d["fuel_basis"]]
    energy = [("Vapor que sale de la caldera (lo que se aprovecha)", boiler["useful_steam_heat_fraction_of_fuel"]),
              ("Purga", boiler["blowdown_heat_fraction_of_fuel"])]
    energy += [(SR.LOSS_ES[k][0], v) for k, v in boiler["losses_fraction_of_heating_value"].items() if abs(v) > 5e-5]
    energy = [energy[0]] + sorted(energy[1:], key=lambda x: -x[1])
    out = {"2.": bar_chart([(c, 100 * v, f"{100 * v:.1f}") for c, v in energy],
                           "A dónde se va la energía del combustible", f"Unidades de cada 100 de energía del combustible (base {basis})")}
    ev = [m for m in d["measures"] if m["status"] == "evaluated"]
    if ev:
        annual = ev[0]["savings"]["annual"] is not None
        rows = sorted(((m["label"], m["savings"]["annual"]["cost"]["total"] if annual else m["savings"]["per_tonne"]["cost"])
                       for m in ev), key=lambda x: -x[1])
        unit = f"{cur} al año" if annual else f"{cur} por tonelada de vapor"
        out["3."] = bar_chart([(c, v, f"{v:,.0f}" if annual else f"{v:,.2f}") for c, v in rows],
                              "Ahorro de cada medida, una por una", f"{unit}. No se suman: ver «Todas las medidas juntas»")
    dist = d.get("distribution")
    if dist and dist["hours_per_year"] is not None:
        rows = [(label, (dist[key] or {}).get("cost_per_year")) for key, label in
                (("leaks", "Fugas de vapor"), ("failed_traps", "Trampas falladas abiertas"), ("bare_pipes", "Tubería sin aislar"))]
        out["Pérdidas"] = bar_chart(sorted(((c, v, f"{v:,.0f}") for c, v in rows if v), key=lambda x: -x[1]),
                                    "Lo que cuestan las pérdidas en la distribución", f"{cur} al año, con el vapor de la línea base")
    return out


def _tiles(d: dict) -> str:
    cur, base = d["currency"], d["baseline"]
    steam, annual, em = base["steam"], base["annual"], base["emissions"]
    tiles = []
    if annual:
        tiles.append((f"{annual['cost']['total']:,.0f} {cur}", "Costo del vapor al año", f"{steam['cost_per_tonne']:,.1f} {cur} por tonelada"))
    else:
        tiles.append((f"{steam['cost_per_tonne']:,.1f} {cur}", "Costo por tonelada de vapor", "sin cifras anuales"))
    tiles.append((f"{100 * base['boiler']['efficiency']:.1f} %", "Eficiencia de la caldera", f"base {SR.BASIS_ES[d['fuel_basis']]}"))
    comb = d.get("combined")
    if comb and comb.get("savings"):
        s = comb["savings"]
        if s["annual"]:
            tiles.append((f"{s['annual']['cost']['total']:,.0f} {cur}", "Ahorro al año con todas las medidas",
                          f"{100 * s['cost_fraction']:.1f} % del costo del vapor"))
        else:
            tiles.append((f"{s['per_tonne']['cost']:,.1f} {cur}", "Ahorro por tonelada con todas las medidas",
                          f"{100 * s['cost_fraction']:.1f} % del costo del vapor"))
    dist = d.get("distribution")
    if dist and dist["total"]["cost_per_year"]:
        tiles.append((f"{dist['total']['cost_per_year']:,.0f} {cur}", "Pérdidas en la distribución al año", "fugas, trampas y tubería"))
    if em["co2_tonne_per_year"] is not None:
        tiles.append((f"{em['co2_tonne_per_year']:,.0f} t", "CO2 directo al año", f"{em['co2_kg_per_tonne_steam']:,.0f} kg por tonelada de vapor"))
    assumed = len(d.get("assumed_values") or [])
    flags = len((d.get("data_review") or {}).get("flags", []))
    tiles.append((f"{assumed}", "Valores supuestos", f"{flags} dato(s) por confirmar" if flags else "ningún dato por confirmar"))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(v)}</div>'
        f'<div class="s">{html.escape(s)}</div></div>' for v, l, s in tiles) + "</div>"


# ----------------------------------------------------------------------------------------------------------- page
def steam_report_html(report: dict, plant_label: str | None = None, intake_notes: list[str] | None = None) -> str:
    """`report`: the dict `steam_diagnostic_report` returns. Returns the complete HTML document as a string."""
    d = report["diagnostic"]
    charts = _charts(d)
    body = []
    for title, block in markdown_to_html(report["markdown"]):
        if title:
            body.append(f"<h2>{_inline(title)}</h2>")
        if title == "Resumen":
            block = _tiles(d) + block
        for key, svg in charts.items():
            if title.startswith(key):
                block = svg + block
        body.append(block)
    if intake_notes:
        body.append("<h2>Notas de la captura de datos</h2><ul>" + "".join(f"<li>{_inline(n)}</li>" for n in intake_notes)
                    + "</ul>")
    name = f"Diagnóstico de vapor{' — ' + plant_label if plant_label else ''}"
    return ("<!doctype html>\n<html lang=\"es\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(name)}</title><style>{CSS}</style></head><body><main>"
            f"<header class=\"top\"><div class=\"brand\">AlchemAI · estudio con números comprobados</div><h1>{html.escape(name)}</h1>"
            "</header>" + "\n".join(body)
            + f"<footer>Generado por AlchemAI el {html.escape(d['inputs']['today'])}. Cada cifra sale de un motor de cálculo; "
              "el texto se arma sin modelo de lenguaje. Este documento no sustituye una auditoría en sitio.</footer>"
            "</main></body></html>\n")


def find_chrome() -> str | None:
    for p in CHROME_PATHS:
        if os.path.exists(p):
            return p
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def html_to_pdf(html_path: str, pdf_path: str, timeout_s: int = 90) -> dict:
    """Print the HTML to PDF with a local Chrome. Returns {'ok', 'path' | 'reason'}; never raises for a missing browser."""
    chrome = find_chrome()
    if chrome is None:
        return {"ok": False, "reason": "no se encontró Chrome, Chromium ni Edge en este equipo: abre el HTML y usa Imprimir → "
                                       "Guardar como PDF"}
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.path.abspath(pdf_path)}",
           "file://" + os.path.abspath(html_path)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"Chrome no terminó en {timeout_s} s"}
    if run.returncode != 0 or not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
        return {"ok": False, "reason": f"Chrome no produjo el PDF (código {run.returncode}): {run.stderr.strip()[-300:]}"}
    return {"ok": True, "path": os.path.abspath(pdf_path)}
