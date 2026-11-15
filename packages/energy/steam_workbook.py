"""Excel intake of the steam diagnostic (bloque M21): the plant fills one workbook in its own units.

`build_template` writes the workbook (Spanish labels, a unit picker per row, a help text per row). `read_workbook` turns a
filled workbook into the arguments of `steam_system_diagnostic`, converting units and reporting every problem at once with
its cell. Nothing is defaulted here: an empty cell is a missing datum and the engine decides what that prevents. The only
value this module computes is the atmospheric pressure from the altitude (1976 standard atmosphere) when the plant does
not give its barometric pressure; it is reported in `notes`.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

SHEET_DATA, SHEET_HELP, SHEET_LEAKS, SHEET_PIPES = "Datos", "Instrucciones", "Fugas y trampas", "Tubería sin aislar"
COL_LABEL, COL_VALUE, COL_UNIT, COL_HELP, COL_KEY = 1, 2, 3, 4, 5
YES, NO = "Sí", "No"
KGF_CM2, PSI, BHP_KG_S = 98066.5, 6894.757293168, 15.65 / 3600.0     # 1 boiler horsepower = 15.65 kg/h from and at 100 °C

# unit families: label shown in the workbook -> function to SI (None marks gauge pressures: they need the site pressure)
TEMP = {"°C": lambda v: v + 273.15, "°F": lambda v: (v - 32.0) / 1.8 + 273.15, "K": lambda v: v}
DELTA_T = {"°C": lambda v: v, "°F": lambda v: v / 1.8}
P_GAUGE = {"kg/cm² man.": KGF_CM2, "bar man.": 1e5, "psig": PSI, "kPa man.": 1e3}
P_ABS = {"kg/cm² abs.": KGF_CM2, "bar abs.": 1e5, "psia": PSI, "kPa abs.": 1e3}
P_ATM = {"kPa": 1e3, "mbar": 100.0, "mmHg": 133.322387415, "psi": PSI}
LENGTH = {"m": 1.0, "ft": 0.3048}
SMALL_LENGTH = {"mm": 1e-3, "pulgadas": 0.0254}
FLOW = {"t/h": 1000.0 / 3600.0, "kg/h": 1.0 / 3600.0, "lb/h": 0.45359237 / 3600.0, "kg/s": 1.0}
CAPACITY = {**FLOW, "CC (caballos caldera)": BHP_KG_S}
PERCENT = {"%": 0.01}
PPM = {"ppm": 1e-6}
ENERGY = {"GJ": 1.0, "MMBtu": 1.055056, "Gcal": 4.1868}
VOLUME = {"m³": 1.0, "ft³": 0.028316846592}
HV_VOLUME = {"MJ/m³": 1e6, "kJ/m³": 1e3, "kcal/m³": 4186.8, "BTU/ft³": 1055.056 / 0.028316846592}
HV_MASS = {"MJ/kg": 1e6, "kcal/kg": 4186.8, "BTU/lb": 1055.056 / 0.45359237}
PRICE_ENERGY = {"por GJ": 1.0, "por MMBtu": 1.0 / 1.055056}
MASS = {"kg": 1.0, "t": 1000.0}
SPEED = {"m/s": 1.0, "km/h": 1.0 / 3.6}
PLAIN = {"": 1.0}
BASIS = {"Superior (PCS)": "HHV", "Inferior (PCI)": "LHV"}
O2_BASIS = {"Seca": "dry", "Húmeda": "wet"}
HEATING = {"Desaireador o tanque calentado con vapor de la caldera": "deaerator_steam",
           "Tanque sin calentador (solo mezcla de condensado y agua de repuesto)": "mixing_only",
           "No cobrar el calentamiento del agua al combustible": "not_charged"}
DEVICE = {"Economizador": "economizer", "Precalentador de aire": "air_preheater"}
BOILER_TYPE = {"Tubos de humo (pirotubular)": "fire_tube", "Tubos de agua (acuotubular)": "water_tube"}
FUELS = ["Gas natural", "Gas LP", "Otro gas", "Diésel", "Combustóleo", "Otro líquido o sólido"]
COMPONENTS = {"comp_CH4": "CH4", "comp_C2H6": "C2H6", "comp_C3H8": "C3H8", "comp_nC4H10": "n-C4H10", "comp_iC4H10": "i-C4H10",
              "comp_CO2": "CO2", "comp_N2": "N2", "comp_H2": "H2"}
ELEMENTS = {"ult_C": "C", "ult_H": "H", "ult_O": "O", "ult_N": "N", "ult_S": "S", "ult_ash": "ash", "ult_moisture": "moisture"}


def _norm(text) -> str:
    """Loose form of a unit or an option, so a hand-typed «kg/cm2 man», «ºc» or «si» is read as the listed one."""
    t = str(text).replace("º", "").replace("°", "").replace("²", "2").replace("³", "3")   # before NFKD turns «º» into «o»
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[\s\.°º]", "", t).lower()


def _match(value, options) -> str | None:
    """The listed option a typed value means, or None. Exact first; then the loose form, only if it is unambiguous."""
    if value in options:
        return value
    hits = [o for o in options if _norm(o) == _norm(value)]
    return hits[0] if len(hits) == 1 else None


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str                 # 'number' | 'choice' | 'text' | 'money'
    units: tuple = ()         # unit labels (number) or options (choice)
    help: str = ""


def _n(key, label, family, help_=""):
    return Field(key, label, "number", tuple(family), help_)


def _c(key, label, options, help_=""):
    return Field(key, label, "choice", tuple(options), help_)


SECTIONS: list[tuple[str, list[Field]]] = [
    ("1. La planta", [
        Field("plant_label", "Nombre de la planta (o un alias)", "text", (), "Aparece en el título del reporte."),
        _c("currency", "Moneda de todos los importes", ["MXN", "USD"], "Todos los precios y recibos de este archivo."),
        _n("altitude", "Altitud de la planta sobre el nivel del mar", LENGTH,
           "Sirve para la presión atmosférica. Si conoces la presión barométrica, llena la fila siguiente."),
        _n("p_atm", "Presión atmosférica del sitio (opcional)", P_ATM, "Si la das, se usa en lugar de la altitud."),
    ]),
    ("2. El combustible", [
        _c("fuel_name", "¿Qué combustible quema la caldera?", FUELS),
        _n("fuel_T", "Temperatura del combustible en el quemador", TEMP, "Si no se mide, la del cuarto de calderas."),
        _n("comp_CH4", "Composición del gas: metano", PERCENT,
           "Porcentaje molar, del análisis del proveedor. Si es gas natural y NO tienes el análisis, deja vacías estas "
           "filas: la composición se ajusta al poder calorífico del recibo y el reporte lo marca como supuesto."),
        _n("comp_C2H6", "   etano", PERCENT), _n("comp_C3H8", "   propano", PERCENT),
        _n("comp_nC4H10", "   butano normal", PERCENT), _n("comp_iC4H10", "   isobutano", PERCENT),
        _n("comp_CO2", "   dióxido de carbono", PERCENT), _n("comp_N2", "   nitrógeno", PERCENT),
        _n("comp_H2", "   hidrógeno", PERCENT),
        _n("hv_value", "Poder calorífico que da el proveedor de gas (opcional)", HV_VOLUME,
           "Solo si no tienes el análisis y el recibo no trae energía y volumen."),
        _c("hv_basis", "   ¿superior o inferior?", BASIS),
        _n("hv_T", "   temperatura base de ese metro cúbico", TEMP), _n("hv_P", "   presión base de ese metro cúbico", P_ABS),
        _n("ult_C", "Combustible líquido o sólido: carbono", PERCENT, "Porcentaje en masa, del análisis elemental."),
        _n("ult_H", "   hidrógeno", PERCENT), _n("ult_O", "   oxígeno", PERCENT), _n("ult_N", "   nitrógeno", PERCENT),
        _n("ult_S", "   azufre", PERCENT), _n("ult_ash", "   ceniza", PERCENT), _n("ult_moisture", "   humedad", PERCENT),
        _n("ult_HHV", "   poder calorífico superior medido", HV_MASS),
    ]),
    ("3. Lo que pagas (un precio, o los datos del recibo)", [
        _n("fuel_price", "Combustible: precio", PRICE_ENERGY, "O llena los datos del recibo de abajo, no los dos."),
        _c("fuel_price_basis", "   ¿por energía superior o inferior?", BASIS),
        Field("fuel_bill_amount", "Combustible: importe del recibo", "money", (), "Con todo lo que incluya el importe."),
        Field("fuel_bill_period", "   periodo del recibo", "text", (), "Por ejemplo «octubre 2026» o «nov 2025 a oct 2026»."),
        _n("fuel_bill_energy", "   energía facturada", ENERGY),
        _c("fuel_bill_basis", "   ¿energía superior o inferior?", BASIS),
        _n("fuel_bill_volume", "   volumen facturado", VOLUME, "Con energía y volumen se obtiene el poder calorífico del gas."),
        _n("fuel_bill_vol_T", "   temperatura base del volumen", TEMP, "Viene en el recibo o en el contrato."),
        _n("fuel_bill_vol_P", "   presión base del volumen", P_ABS, "Viene en el recibo o en el contrato."),
        _n("fuel_bill_mass", "   masa facturada (líquidos o gas LP)", MASS),
        _c("fuel_bill_is_year", "   ¿El recibo cubre un año completo de operación y SOLO esta caldera?", [YES, NO],
           "Si es Sí, se toma como el consumo anual de combustible."),
        Field("water_price", "Agua de repuesto: precio por m³", "money", (),
              "Incluye el tratamiento si quieres cobrarlo. Si no lo sabes, déjalo vacío: el costo del vapor saldrá sin agua."),
        Field("water_bill_amount", "Agua: importe del recibo", "money"),
        _n("water_bill_volume", "   volumen facturado", {"m³": 1.0}),
        Field("water_bill_period", "   periodo del recibo", "text"),
        Field("elec_price", "Electricidad: precio por kWh", "money", (), "Solo cuenta si das el consumo de la caldera."),
        Field("elec_bill_amount", "Electricidad: importe del recibo", "money"),
        _n("elec_bill_energy", "   energía facturada", {"kWh": 1.0}),
        Field("elec_bill_period", "   periodo del recibo", "text"),
        _n("aux_power", "Electricidad de ventiladores y bombas de la caldera", {"kWh por tonelada de vapor generado": 1.0},
           "Opcional. Sin este dato el costo del vapor no incluye electricidad."),
    ]),
    ("4. La caldera", [
        _n("steam_pressure", "Presión de operación", {**P_GAUGE, **P_ABS}, "La del manómetro de la caldera."),
        _n("superheat", "Sobrecalentamiento (0 si el vapor es saturado)", DELTA_T),
        _n("blowdown", "Purga, del agua de alimentación", PERCENT, "0 solo si de verdad no hay purga."),
        _n("rated_capacity", "Capacidad nominal (placa)", CAPACITY, "Opcional. Sirve para saber a qué carga trabaja."),
        _c("boiler_type", "Tipo de caldera", BOILER_TYPE,
           "Opcional. Si no conoces la pérdida por paredes, con el tipo y la capacidad se elige la fuente publicada."),
    ]),
    ("5. La combustión (de la última prueba de gases)", [
        _n("o2", "Oxígeno en la chimenea", PERCENT, "O da el exceso de aire en la fila de abajo, no los dos."),
        _c("o2_basis", "   ¿base seca o húmeda?", O2_BASIS, "Los analizadores portátiles miden en base seca."),
        _n("excess_air", "Exceso de aire (si no tienes el oxígeno)", PERCENT),
        _n("co", "Monóxido de carbono en chimenea, base seca (opcional)", PPM),
        _n("stack_T", "Temperatura de los gases en la chimenea", TEMP, "A la misma carga que el oxígeno."),
        _n("air_T", "Temperatura del aire que entra al quemador", TEMP),
        _n("air_RH", "Humedad relativa del aire (opcional)", PERCENT, "Sin este dato se calcula con aire seco y se avisa."),
        _n("setting_loss", "Pérdida por paredes y radiación de la caldera", PERCENT,
           "De la hoja del fabricante o de una prueba. Si no la conoces, deja vacío y marca Sí en la fila «No la conozco»."),
        _c("setting_loss_basis", "   ¿del poder calorífico superior o inferior?", BASIS),
        _c("setting_loss_unknown", "   No la conozco", [YES, NO],
           "Con Sí se usa un valor publicado (guía del Departamento de Energía de EE. UU., o la tabla del fabricante "
           "Cleaver-Brooks si es de tubos de humo y das capacidad y producción) y el reporte lo marca como supuesto."),
    ]),
    ("6. El agua", [
        _c("fw_mode", "¿Cómo se calienta el agua de alimentación?", HEATING),
        _n("fw_T", "Temperatura del agua que entra a la caldera", TEMP, "No hace falta si el tanque no tiene calentador."),
        _n("makeup_T", "Temperatura del agua de repuesto", TEMP),
        _n("cond_return", "Condensado que regresa, del vapor entregado", PERCENT, "0 si no regresa nada."),
        _n("cond_T", "Temperatura del condensado que regresa", TEMP),
    ]),
    ("7. La operación", [
        _n("hours", "Horas de operación al año", {"h al año": 1.0}),
        _n("steam_production", "Producción promedio de vapor de la caldera", FLOW, "O da el consumo anual de combustible."),
        _n("fuel_year_energy", "Consumo anual de combustible, en energía", ENERGY,
           "Si das producción y consumo, las cifras anuales salen de la producción y el consumo sirve de comprobación."),
        _c("fuel_year_basis", "   ¿superior o inferior?", BASIS),
        _n("fuel_year_volume", "Consumo anual de combustible, en volumen", VOLUME, "A las condiciones base del recibo."),
    ]),
    ("8. Medidas de ahorro (llena solo las que quieras evaluar)", [
        _n("m1_o2_target", "Afinar la combustión: oxígeno objetivo en chimenea (misma base que hoy)", PERCENT,
           "Lo da el proveedor del quemador."),
        _c("m1_published", "   No tengo objetivo: usar el publicado (10 % de exceso de aire, solo gas natural)", [YES, NO]),
        _n("m1_stack_after", "   temperatura de chimenea después de afinar (opcional)", TEMP),
        Field("m1_investment", "   inversión cotizada (opcional)", "money"),
        _n("m2_stack_target", "Economizador o precalentador: temperatura de chimenea a la salida del equipo", TEMP),
        _c("m2_published", "   No tengo objetivo: usar el publicado (121 °C, solo gas natural)", [YES, NO]),
        _c("m2_device", "   ¿qué equipo es?", DEVICE),
        Field("m2_investment", "   inversión cotizada (opcional)", "money"),
        _n("m3_blowdown_target", "Bajar la purga: purga objetivo, del agua de alimentación", PERCENT,
           "La da el proveedor del tratamiento de agua."),
        Field("m3_investment", "   inversión cotizada (opcional)", "money"),
        _n("m4_flash_pressure", "Recuperar calor de la purga: presión del tanque de flasheo", {**P_GAUGE, **P_ABS}),
        _n("m4_drain_T", "   temperatura a la que la purga se va al drenaje", TEMP),
        Field("m4_investment", "   inversión cotizada (opcional)", "money"),
        _n("m5_return_target", "Regresar más condensado: retorno objetivo, del vapor entregado", PERCENT),
        _n("m5_cond_T_after", "   temperatura del condensado después (opcional)", TEMP),
        Field("m5_investment", "   inversión cotizada (opcional)", "money"),
    ]),
]
SECTIONS.append(("9. Recorrido de la distribución (solo si llenas las hojas «Fugas y trampas» o «Tubería sin aislar»)", [
    _n("ambient_T", "Temperatura del aire junto a las tuberías sin aislar", TEMP),
    _n("wind", "Viento sobre esas tuberías (0 si están bajo techo)", SPEED),
]))
FIELDS = {f.key: f for _, fs in SECTIONS for f in fs}
# a sub-row («   etano») only makes sense next to the row it hangs from: full labels for the copy shown in the report
FULL_LABEL: dict[str, str] = {}
for _title, _fields in SECTIONS:
    _parent = ""
    for _f in _fields:
        if _f.label.startswith(" "):
            FULL_LABEL[_f.key] = f"{_parent.split(':')[0]} — {_f.label.strip()}" if _parent else _f.label.strip()
        else:
            _parent = _f.label
            FULL_LABEL[_f.key] = _f.label
INSTRUCTIONS = [
    "Diagnóstico de vapor: datos de la planta",
    "",
    "1. Llena la hoja «Datos»: un valor por fila en la columna «Valor» y su unidad en la columna «Unidad» (lista).",
    "2. Lo que no sepas, déjalo vacío. No pongas un número aproximado: el reporte dirá qué faltó y qué impide.",
    "3. Dos datos casi nunca se conocen y tienen salida:",
    "     - la composición del gas natural: deja vacías sus filas y da el recibo con energía y volumen;",
    "     - la pérdida por paredes de la caldera: marca «No la conozco».",
    "4. Las medidas de ahorro son opcionales. Sin un objetivo, una medida sale como «no evaluada» y se dice qué hace falta.",
    "5. Las hojas «Fugas y trampas» y «Tubería sin aislar» son opcionales: una fila por grupo.",
    "6. Las presiones del manómetro son manométricas («man.»). Si tu dato es absoluto, elige la unidad «abs.».",
    "",
    "Los datos de este archivo no se publican. El reporte dice de dónde sale cada número.",
]
LEAK_COLUMNS = [("Tipo", ["Fuga de vapor", "Trampa de vapor fallada (abierta)"]), ("Descripción (opcional)", None),
                ("Cantidad", None), ("Diámetro del orificio", list(SMALL_LENGTH)), ("Presión del vapor", list({**P_GAUGE, **P_ABS})),
                ("Descarga a (vacío = a la atmósfera)", list({**P_GAUGE, **P_ABS}))]
PIPE_COLUMNS = [("Descripción (opcional)", None), ("Diámetro exterior del tubo", list(SMALL_LENGTH)), ("Longitud", list(LENGTH)),
                ("Presión del vapor que lleva", list({**P_GAUGE, **P_ABS})),
                ("Espesor del aislamiento propuesto", list(SMALL_LENGTH)),
                ("Conductividad del aislamiento (ficha del proveedor)", ["W/m·K"])]


# ------------------------------------------------------------------------------------------------------ template
def build_template(path: str, example: dict | None = None) -> str:
    """Write the workbook. `example`: {field key: (value, unit or None)} to pre-fill it (used for the illustrative file)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_HELP
    for i, line in enumerate(INSTRUCTIONS, start=1):
        ws.cell(row=i, column=1, value=line).font = Font(bold=(i == 1), size=14 if i == 1 else 11)
    ws.column_dimensions["A"].width = 120

    wd = wb.create_sheet(SHEET_DATA)
    head = PatternFill("solid", fgColor="1F3A5F")
    band = PatternFill("solid", fgColor="DCE6F1")
    entry = PatternFill("solid", fgColor="FFF8DC")
    for col, text in ((COL_LABEL, "Dato"), (COL_VALUE, "Valor"), (COL_UNIT, "Unidad"), (COL_HELP, "Ayuda"), (COL_KEY, "clave")):
        c = wd.cell(row=1, column=col, value=text)
        c.font, c.fill = Font(bold=True, color="FFFFFF"), head
    row = 2
    for title, fields in SECTIONS:
        c = wd.cell(row=row, column=COL_LABEL, value=title)
        c.font = Font(bold=True)
        for col in range(COL_LABEL, COL_HELP + 1):
            wd.cell(row=row, column=col).fill = band
        row += 1
        for f in fields:
            wd.cell(row=row, column=COL_LABEL, value=f.label)
            v = wd.cell(row=row, column=COL_VALUE)
            v.fill = entry
            wd.cell(row=row, column=COL_HELP, value=f.help).alignment = Alignment(wrap_text=True, vertical="top")
            wd.cell(row=row, column=COL_KEY, value=f.key)
            if f.kind == "choice":
                dv = DataValidation(type="list", formula1='"' + ",".join(f.units) + '"', allow_blank=True)
                wd.add_data_validation(dv)
                dv.add(v)
            elif f.kind == "number":
                u = wd.cell(row=row, column=COL_UNIT, value=f.units[0])
                if len(f.units) > 1:
                    dv = DataValidation(type="list", formula1='"' + ",".join(f.units) + '"', allow_blank=False)
                    wd.add_data_validation(dv)
                    dv.add(u)
            elif f.kind == "money":
                wd.cell(row=row, column=COL_UNIT, value="en la moneda elegida")
            if example and f.key in example:
                value, unit = example[f.key]
                v.value = value
                if unit is not None:
                    wd.cell(row=row, column=COL_UNIT, value=unit)
            row += 1
    for letter, width in (("A", 74), ("B", 22), ("C", 36), ("D", 90)):
        wd.column_dimensions[letter].width = width
    wd.column_dimensions["E"].hidden = True
    wd.freeze_panes = "B2"

    for name, columns, rows in ((SHEET_LEAKS, LEAK_COLUMNS, (example or {}).get("__leaks__", [])),
                                (SHEET_PIPES, PIPE_COLUMNS, (example or {}).get("__pipes__", []))):
        wt = wb.create_sheet(name)
        col = 1
        for text, units in columns:
            c = wt.cell(row=1, column=col, value=text)
            c.font, c.fill = Font(bold=True, color="FFFFFF"), head
            wt.column_dimensions[c.column_letter].width = 34
            if units and text != "Tipo":
                u = wt.cell(row=1, column=col + 1, value="Unidad")
                u.font, u.fill = Font(bold=True, color="FFFFFF"), head
                dv = DataValidation(type="list", formula1='"' + ",".join(units) + '"', allow_blank=True)
                wt.add_data_validation(dv)
                dv.add(f"{u.column_letter}2:{u.column_letter}60")
                col += 1
            elif units:
                dv = DataValidation(type="list", formula1='"' + ",".join(units) + '"', allow_blank=True)
                wt.add_data_validation(dv)
                dv.add(f"{c.column_letter}2:{c.column_letter}60")
            col += 1
        for r, values in enumerate(rows, start=2):
            for cidx, value in enumerate(values, start=1):
                wt.cell(row=r, column=cidx, value=value)
        wt.freeze_panes = "A2"
    # printing: a plant may fill the sheet by hand. Landscape, every column on one page width, header row repeated
    from openpyxl.worksheet.properties import PageSetupProperties
    for sheet in wb.worksheets:
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth, sheet.page_setup.fitToHeight = 1, 0
        sheet.page_margins.left = sheet.page_margins.right = 0.4
        sheet.page_margins.top = sheet.page_margins.bottom = 0.5
        if sheet.title != SHEET_HELP:
            sheet.print_title_rows = "1:1"
    wb.active = wb.index(wd)
    wb.save(path)
    _fix_font_order(path)
    return path


# children of <font> in the order the SpreadsheetML schema (CT_Font) requires
FONT_CHILD_ORDER = ["b", "i", "strike", "condense", "extend", "outline", "shadow", "u", "vertAlign", "sz", "color", "name",
                    "family", "charset", "scheme"]


def _fix_font_order(path: str) -> None:
    """openpyxl writes the children of the default <font> in an order the Open XML schema does not allow (Excel tolerates
    it; Microsoft's validator does not). Rewrite xl/styles.xml with the schema order so the file validates cleanly."""
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ET.register_namespace("", ns)
    rank = {name: i for i, name in enumerate(FONT_CHILD_ORDER)}
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/styles.xml":
                root = ET.fromstring(data)
                for font in root.iter(f"{{{ns}}}font"):
                    children = sorted(list(font), key=lambda e: rank.get(e.tag.split("}")[1], len(rank)))
                    for child in list(font):
                        font.remove(child)
                    font.extend(children)
                data = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(root, encoding="utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)


# -------------------------------------------------------------------------------------------------------- reader
class _Sheet:
    """Values of the «Datos» sheet by field key, with the cell of each for error messages."""

    def __init__(self, ws):
        self.problems: list[str] = []
        self.raw: dict[str, tuple] = {}
        for r in range(2, ws.max_row + 1):
            key = ws.cell(row=r, column=COL_KEY).value
            if key in FIELDS:
                cell = ws.cell(row=r, column=COL_VALUE)
                value, unit = cell.value, ws.cell(row=r, column=COL_UNIT).value
                if isinstance(value, str) and not value.strip():
                    value = None
                # typing «8%» in a spreadsheet stores 0.08 with a percent format: read it back as the 8 the plant meant
                if isinstance(value, (int, float)) and not isinstance(value, bool) and "%" in (cell.number_format or "") \
                        and unit == "%":
                    value = round(value * 100.0, 10)
                self.raw[key] = (value, unit, f"{SHEET_DATA}!B{r}")
        lost = sorted(k for k in FIELDS if k not in self.raw)
        if lost:
            self.problems.append("Al archivo le faltan filas de la plantilla (" + ", ".join(lost[:6])
                                 + ("…" if len(lost) > 6 else "") + "). Usa la plantilla sin borrar filas.")

    def has(self, key: str) -> bool:
        return self.raw.get(key, (None,))[0] is not None

    def where(self, key: str) -> str:
        return f"{self.raw[key][2]} («{FIELDS[key].label.strip()}»)" if key in self.raw else key

    def _number(self, key: str):
        value, unit, _ = self.raw.get(key, (None, None, None))
        if value is None:
            return None, None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            text = str(value).replace("%", "").replace(" ", "").strip()
            if "," in text and not re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):
                self.problems.append(f"{self.where(key)}: «{value}» no es un número. Usa punto decimal (por ejemplo 0.045) y, "
                                     "si separas los miles, que sea con coma (1,500).")
                return None, None
            try:
                value = float(text.replace(",", ""))
            except ValueError:
                self.problems.append(f"{self.where(key)}: «{value}» no es un número.")
                return None, None
        return float(value), unit

    def number(self, key: str):
        """SI value of a numeric field, or None if empty. Gauge pressures come back as ('gauge', Pa)."""
        value, unit = self._number(key)
        if value is None:
            return None
        f = FIELDS[key]
        listed = _match(unit, f.units) if unit is not None else None
        if listed is None:
            self.problems.append(f"{self.where(key)}: elige la unidad de la lista ({', '.join(f.units)}); se leyó «{unit}».")
            return None
        unit = listed
        for family in (TEMP, DELTA_T):
            if f.units == tuple(family):
                return family[unit](value)
        if unit in P_GAUGE and set(f.units) >= set(P_GAUGE):
            return ("gauge", value * P_GAUGE[unit])
        for family in (P_ABS, P_ATM, LENGTH, CAPACITY, PERCENT, PPM, ENERGY, VOLUME, HV_VOLUME, HV_MASS, PRICE_ENERGY, MASS,
                       SPEED):
            if unit in family and set(f.units) <= set(family) | set(P_GAUGE):
                return value * family[unit]
        return value                                                             # single-unit rows (h, kWh, m³, kWh/t)

    def money(self, key: str):
        value, _ = self._number(key)
        return value

    def choice(self, key: str, mapping: dict | None = None):
        value = self.raw.get(key, (None,))[0]
        if value is None:
            return None
        options = FIELDS[key].units
        listed = _match(value, options)
        if listed is None:
            self.problems.append(f"{self.where(key)}: elige una opción de la lista ({'; '.join(options)}); se leyó «{value}».")
            return None
        return mapping[listed] if mapping else listed

    def text(self, key: str):
        value = self.raw.get(key, (None,))[0]
        return None if value is None else str(value).strip()


def standard_atmosphere_Pa(altitude_m: float) -> float:
    from packages.energy.steam_diagnostic import standard_atmosphere_Pa as engine

    return engine(altitude_m)


def _table_rows(ws, n_cols: int) -> list[tuple[int, list]]:
    rows = []
    for r in range(2, ws.max_row + 1):
        values = [ws.cell(row=r, column=c).value for c in range(1, n_cols + 1)]
        if any(v is not None and str(v).strip() != "" for v in values):
            rows.append((r, values))
    return rows


def read_workbook(path: str) -> dict:
    """Filled workbook -> {'inputs': kwargs of steam_system_diagnostic, 'plant_label', 'notes'}.

    Raises ValueError listing every problem of the file (wrong unit, text where a number goes, contradictory rows).
    Missing data are NOT problems here: the engine refuses or degrades and says what each one prevents.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    if SHEET_DATA not in wb.sheetnames:
        raise ValueError(f"El archivo no tiene la hoja «{SHEET_DATA}»: usa la plantilla del diagnóstico de vapor.")
    s = _Sheet(wb[SHEET_DATA])
    notes: list[str] = []
    P = s.problems

    # ---- site pressure
    p_atm, alt = s.number("p_atm"), s.number("altitude")
    from_altitude = p_atm is None and alt is not None
    if from_altitude:
        p_atm = standard_atmosphere_Pa(alt)                                      # the engine recomputes and labels it

    def absolute(key: str):
        v = s.number(key)
        if isinstance(v, tuple):
            if p_atm is None:
                P.append(f"{s.where(key)}: es una presión manométrica y falta la altitud o la presión atmosférica del sitio "
                         "(sección 1) para pasarla a absoluta.")
                return None
            return v[1] + p_atm
        return v

    # ---- fuel
    fuel_name = s.choice("fuel_name")
    comp = {COMPONENTS[k]: s.number(k) for k in COMPONENTS if s.has(k)}
    comp = {k: v for k, v in comp.items() if v}
    ult = {ELEMENTS[k]: s.number(k) for k in ELEMENTS if s.has(k)}
    fuel = None
    if comp and ult:
        P.append("Sección 2: llenaste composición de gas y análisis elemental. Llena solo el que corresponde a tu combustible.")
    elif comp:
        total = sum(comp.values())
        if abs(total - 1.0) > 0.005:
            P.append(f"Sección 2: la composición del gas suma {100 * total:.2f} %, no 100 %. Revisa el análisis.")
        fuel = {"type": "gas", "composition": {k: v / total for k, v in comp.items()}, "T_K": s.number("fuel_T")}
    elif ult:
        fuel = {"type": "ultimate", "mass_fractions": {k: v for k, v in ult.items() if v is not None},
                "HHV_J_kg": s.number("ult_HHV"), "T_K": s.number("fuel_T")}
    elif fuel_name == "Gas natural":
        fuel = {"type": "natural_gas_unknown", "T_K": s.number("fuel_T")}
        if s.has("hv_value"):
            fuel["heating_value"] = {"value_J_m3": s.number("hv_value"), "basis": s.choice("hv_basis", BASIS),
                                     "volume_reference": {"T_K": s.number("hv_T"), "P_Pa": s.number("hv_P")}}
    elif fuel_name is not None:
        P.append(f"Sección 2: para «{fuel_name}» hace falta la composición (gases) o el análisis elemental con el poder "
                 "calorífico (líquidos y sólidos). Solo el gas natural puede ajustarse al poder calorífico del recibo.")

    # ---- prices
    def one_of(price_key: str, bill_key: str, what: str):
        if s.has(price_key) and s.has(bill_key):
            P.append(f"Sección 3: para {what} diste precio y también recibo. Da solo uno de los dos.")
        return s.has(price_key), s.has(bill_key)

    fuel_cost = water_cost = electricity_cost = None
    has_price, has_bill = one_of("fuel_price", "fuel_bill_amount", "el combustible")
    if has_price:
        fuel_cost = {"price_per_GJ": s.number("fuel_price"), "basis": s.choice("fuel_price_basis", BASIS)}
    elif has_bill:
        bill = {"amount": s.money("fuel_bill_amount"), "period": s.text("fuel_bill_period")}
        if s.has("fuel_bill_energy"):
            bill.update(energy_GJ=s.number("fuel_bill_energy"), basis=s.choice("fuel_bill_basis", BASIS))
        if s.has("fuel_bill_volume"):
            bill.update(volume_m3=s.number("fuel_bill_volume"),
                        volume_reference={"T_K": s.number("fuel_bill_vol_T"), "P_Pa": s.number("fuel_bill_vol_P")})
        if s.has("fuel_bill_mass"):
            bill["mass_kg"] = s.number("fuel_bill_mass")
        fuel_cost = {"bill": bill}
    has_price, has_bill = one_of("water_price", "water_bill_amount", "el agua")
    if has_price:
        water_cost = {"price_per_m3": s.money("water_price")}
    elif has_bill:
        water_cost = {"bill": {"amount": s.money("water_bill_amount"), "volume_m3": s.number("water_bill_volume"),
                               "period": s.text("water_bill_period")}}
    has_price, has_bill = one_of("elec_price", "elec_bill_amount", "la electricidad")
    if has_price:
        electricity_cost = {"price_per_kWh": s.money("elec_price")}
    elif has_bill:
        electricity_cost = {"bill": {"amount": s.money("elec_bill_amount"), "energy_kWh": s.number("elec_bill_energy"),
                                     "period": s.text("elec_bill_period")}}

    # ---- boiler, combustion, water, operation
    def keep(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    boiler = keep({"steam_pressure_Pa": absolute("steam_pressure"), "superheat_K": s.number("superheat"),
                   "blowdown_fraction": s.number("blowdown"), "auxiliary_power_kWh_per_tonne_generated": s.number("aux_power"),
                   "rated_steam_capacity_kg_s": s.number("rated_capacity"),
                   "boiler_type": s.choice("boiler_type", BOILER_TYPE)})
    combustion = keep({"site_altitude_m": alt if from_altitude else None, "site_pressure_Pa": None if from_altitude else p_atm,
                       "stack_T_K": s.number("stack_T"),
                       "excess_air_fraction": s.number("excess_air"), "flue_CO_dry_mole_fraction": s.number("co"),
                       "setting_loss_fraction": s.number("setting_loss"),
                       "setting_loss_basis": s.choice("setting_loss_basis", BASIS) if s.has("setting_loss") else None})
    if s.has("o2"):
        combustion["flue_O2"] = {"mole_fraction": s.number("o2"), "basis": s.choice("o2_basis", O2_BASIS)}
        if combustion["flue_O2"]["basis"] is None:
            P.append(f"{s.where('o2_basis')}: di si el oxígeno se midió en base seca o húmeda.")
    if s.has("air_T"):
        combustion["air"] = keep({"T_K": s.number("air_T"), "relative_humidity": s.number("air_RH")})
    if s.choice("setting_loss_unknown") == YES:
        if s.has("setting_loss"):
            P.append("Sección 5: diste la pérdida por paredes y también marcaste «No la conozco». Deja solo una.")
        else:
            combustion["setting_loss_unknown"] = True
    water = keep({"feedwater_heating": s.choice("fw_mode", HEATING), "feedwater_T_K": s.number("fw_T"),
                  "makeup_water_T_K": s.number("makeup_T"), "condensate_return_fraction": s.number("cond_return"),
                  "condensate_T_K": s.number("cond_T")})
    operation = keep({"hours_per_year": s.number("hours"), "steam_generated_kg_s": s.number("steam_production")})
    if s.has("fuel_year_energy") and s.has("fuel_year_volume"):
        P.append("Sección 7: da el consumo anual de combustible en energía o en volumen, no los dos.")
    elif s.has("fuel_year_energy"):
        operation["fuel_consumption"] = {"energy_GJ_per_year": s.number("fuel_year_energy"),
                                         "basis": s.choice("fuel_year_basis", BASIS)}
    elif s.has("fuel_year_volume"):
        operation["fuel_consumption"] = {"volume_m3_per_year": s.number("fuel_year_volume"),
                                         "volume_reference": {"T_K": s.number("fuel_bill_vol_T"),
                                                              "P_Pa": s.number("fuel_bill_vol_P")}}
    elif s.choice("fuel_bill_is_year") == YES and fuel_cost and "bill" in fuel_cost:
        operation["fuel_consumption"] = {"from_fuel_bill": True}
    if "fuel_consumption" in operation and "steam_generated_kg_s" in operation:
        # both given: the annual figures follow the steam meter and the fuel consumption becomes the cross-check
        operation["annual_basis"] = "steam"

    # ---- measures
    measures: dict = {}
    o2_basis = (combustion.get("flue_O2") or {}).get("basis")
    m1 = keep({"stack_T_K_after": s.number("m1_stack_after"), "investment": s.money("m1_investment")})
    if s.has("m1_o2_target"):
        if o2_basis is None:
            P.append(f"{s.where('m1_o2_target')}: el oxígeno objetivo necesita el oxígeno de hoy y su base (sección 5).")
        m1["target_flue_O2"] = {"mole_fraction": s.number("m1_o2_target"), "basis": o2_basis}
    if s.choice("m1_published") == YES:
        if s.has("m1_o2_target"):
            P.append("Sección 8, afinar la combustión: diste un objetivo y también pediste el publicado. Deja solo uno.")
        m1["use_published_target"] = True
    if "target_flue_O2" in m1 or "use_published_target" in m1:
        measures["reduce_excess_air"] = m1
    m2 = keep({"target_stack_T_K": s.number("m2_stack_target"), "device": s.choice("m2_device", DEVICE),
               "investment": s.money("m2_investment")})
    if s.choice("m2_published") == YES:
        if s.has("m2_stack_target"):
            P.append("Sección 8, economizador: diste un objetivo y también pediste el publicado. Deja solo uno.")
        m2["use_published_target"] = True
    if "target_stack_T_K" in m2 or "use_published_target" in m2:
        measures["reduce_stack_temperature"] = m2
    if s.has("m3_blowdown_target"):
        measures["reduce_blowdown"] = keep({"target_blowdown_fraction": s.number("m3_blowdown_target"),
                                            "investment": s.money("m3_investment")})
    if s.has("m4_flash_pressure") or s.has("m4_drain_T"):
        measures["recover_blowdown_heat"] = keep({"flash_pressure_Pa": absolute("m4_flash_pressure"),
                                                  "drain_T_K": s.number("m4_drain_T"), "investment": s.money("m4_investment")})
    if s.has("m5_return_target"):
        measures["increase_condensate_return"] = keep({"target_condensate_return_fraction": s.number("m5_return_target"),
                                                       "condensate_T_K_after": s.number("m5_cond_T_after"),
                                                       "investment": s.money("m5_investment")})

    # ---- distribution tables (optional sheets)
    distribution = _read_distribution(wb, p_atm, P)
    if distribution.get("bare_pipes"):
        distribution.update(keep({"ambient_T_K": s.number("ambient_T"), "wind_speed_m_s": s.number("wind")}))
    distribution = {k: v for k, v in distribution.items() if v not in ([], None)}

    if P:
        raise ValueError("El archivo tiene datos que hay que corregir antes de calcular:\n- " + "\n- ".join(P))
    currency = s.choice("currency")
    inputs = keep({"currency": currency, "fuel": fuel, "combustion": combustion or None, "boiler": boiler or None,
                   "water": water or None, "fuel_cost": fuel_cost, "water_cost": water_cost,
                   "electricity_cost": electricity_cost, "operation": operation or None, "measures": measures or None,
                   "distribution": distribution or None})
    as_entered = [[FULL_LABEL[k], v if not isinstance(v, float) else f"{v:g}", u or ""]
                  for k, (v, u, _) in s.raw.items() if v is not None and k != "plant_label"]
    return {"inputs": inputs, "plant_label": s.text("plant_label"), "notes": _suspects(s) + notes,
            "missing_essentials": _essentials(s), "as_entered": as_entered}


def _essentials(s: _Sheet) -> list[str]:
    """Rows without which the engine refuses, named as the plant sees them (cell and label). The engine stays the judge."""
    need: list[str] = []

    def row(key: str) -> None:
        if not s.has(key):
            need.append(s.where(key))

    def either(a: str, b: str, text: str) -> None:
        if not s.has(a) and not s.has(b):
            need.append(f"{s.where(a)} o {s.where(b)}: {text}")

    row("currency")
    either("altitude", "p_atm", "la altitud o la presión atmosférica")
    row("fuel_name")
    row("fuel_T")
    either("fuel_price", "fuel_bill_amount", "el precio del combustible o el importe del recibo")
    if s.has("fuel_bill_amount") and not any(s.has(k) for k in ("fuel_bill_energy", "fuel_bill_volume", "fuel_bill_mass")):
        need.append(f"{s.where('fuel_bill_energy')}: la energía, el volumen o la masa que factura el recibo")
    for key in ("steam_pressure", "blowdown"):
        row(key)
    either("o2", "excess_air", "el oxígeno en chimenea o el exceso de aire")
    for key in ("stack_T", "air_T"):
        row(key)
    if not s.has("setting_loss") and _match(s.raw.get("setting_loss_unknown", (None,))[0], [YES, NO]) != YES:
        need.append(f"{s.where('setting_loss')}: la pérdida por paredes, o marcar Sí en «No la conozco»")
    row("fw_mode")
    mode = _match(s.raw.get("fw_mode", (None,))[0], list(HEATING))
    if mode is not None and HEATING.get(mode) != "mixing_only":
        row("fw_T")
    for key in ("makeup_T", "cond_return"):
        row(key)
    back = s.raw.get("cond_return", (None,))[0]
    if isinstance(back, (int, float)) and back > 0 and HEATING.get(mode) != "not_charged":
        row("cond_T")
    return need


# percentages a plant sometimes types as a fraction (0.04 for 4 %): below 1 % they are flagged, never changed
FRACTION_SUSPECTS = ("blowdown", "o2", "air_RH", "cond_return", "excess_air", "m1_o2_target", "m3_blowdown_target",
                     "m5_return_target", "comp_CH4")


def _suspects(s: _Sheet) -> list[str]:
    out = []
    for key in FRACTION_SUSPECTS:
        value = s.raw.get(key, (None,))[0]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 1:
            out.append(f"REVISAR DATO: {s.where(key)} dice {value:g} %. Esa fila va en porcentaje: si quisiste decir "
                       f"{100 * value:g} %, corrige la celda y vuelve a correr.")
    return out


def _read_distribution(wb, p_atm, problems: list[str]) -> dict:
    """Rows of the two optional sheets, in SI. Model constants are not asked of the plant: the engine names its own."""
    out: dict = {"leaks": [], "failed_traps": [], "bare_pipes": []}

    def cell_number(sheet: str, r: int, c: int, value, what: str):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(str(value).replace(",", "")) if not isinstance(value, (int, float)) else float(value)
        except ValueError:
            problems.append(f"{sheet}, fila {r}: «{value}» en «{what}» no es un número.")
            return None

    def pressure(sheet: str, r: int, value, unit, what: str):
        if value is None:
            return None
        unit = _match(unit, [*P_ABS, *P_GAUGE]) if unit is not None else None
        if unit in P_ABS:
            return value * P_ABS[unit]
        if unit in P_GAUGE:
            if p_atm is None:
                problems.append(f"{sheet}, fila {r}: «{what}» es manométrica y falta la altitud o la presión atmosférica.")
                return None
            return value * P_GAUGE[unit] + p_atm
        problems.append(f"{sheet}, fila {r}: elige la unidad de «{what}» de la lista.")
        return None

    def length(sheet: str, r: int, value, unit, table: dict, what: str):
        if value is None:
            return None
        unit = _match(unit, list(table)) if unit is not None else None
        if unit not in table:
            problems.append(f"{sheet}, fila {r}: elige la unidad de «{what}» de la lista ({', '.join(table)}).")
            return None
        return value * table[unit]

    if SHEET_LEAKS in wb.sheetnames:
        for r, v in _table_rows(wb[SHEET_LEAKS], 9):
            kind, desc = (_match(v[0], LEAK_COLUMNS[0][1]) if v[0] is not None else None), v[1]
            count = cell_number(SHEET_LEAKS, r, 3, v[2], "Cantidad")
            d = length(SHEET_LEAKS, r, cell_number(SHEET_LEAKS, r, 4, v[3], "Diámetro del orificio"), v[4], SMALL_LENGTH,
                       "Diámetro del orificio")
            p = pressure(SHEET_LEAKS, r, cell_number(SHEET_LEAKS, r, 6, v[5], "Presión del vapor"), v[6], "Presión del vapor")
            back = pressure(SHEET_LEAKS, r, cell_number(SHEET_LEAKS, r, 8, v[7], "Descarga a"), v[8], "Descarga a")
            if kind not in LEAK_COLUMNS[0][1]:
                problems.append(f"{SHEET_LEAKS}, fila {r}: elige el tipo de la lista ({'; '.join(LEAK_COLUMNS[0][1])}).")
                continue
            if count is None or d is None or p is None:
                problems.append(f"{SHEET_LEAKS}, fila {r}: faltan la cantidad, el diámetro del orificio o la presión del vapor.")
                continue
            row = {k2: v2 for k2, v2 in {"description": desc, "count": count, "orifice_diameter_m": d, "steam_pressure_Pa": p,
                                         "back_pressure_Pa": back}.items() if v2 is not None}
            out["leaks" if kind == LEAK_COLUMNS[0][1][0] else "failed_traps"].append(row)
    if SHEET_PIPES in wb.sheetnames:
        for r, v in _table_rows(wb[SHEET_PIPES], 11):
            d = length(SHEET_PIPES, r, cell_number(SHEET_PIPES, r, 2, v[1], "Diámetro exterior"), v[2], SMALL_LENGTH,
                       "Diámetro exterior del tubo")
            L = length(SHEET_PIPES, r, cell_number(SHEET_PIPES, r, 4, v[3], "Longitud"), v[4], LENGTH, "Longitud")
            p = pressure(SHEET_PIPES, r, cell_number(SHEET_PIPES, r, 6, v[5], "Presión del vapor"), v[6], "Presión del vapor")
            t = length(SHEET_PIPES, r, cell_number(SHEET_PIPES, r, 8, v[7], "Espesor"), v[8], SMALL_LENGTH,
                       "Espesor del aislamiento propuesto")
            if d is None or L is None or p is None:
                problems.append(f"{SHEET_PIPES}, fila {r}: faltan el diámetro, la longitud o la presión del vapor.")
                continue
            k = cell_number(SHEET_PIPES, r, 10, v[9], "Conductividad del aislamiento")
            if (t is None) != (k is None):
                problems.append(f"{SHEET_PIPES}, fila {r}: para evaluar el aislamiento da el espesor y la conductividad; con "
                                "solo uno no se puede. Deja los dos vacíos para ver solo lo que el tramo pierde hoy.")
            out["bare_pipes"].append({k2: v2 for k2, v2 in {
                "description": v[0], "outer_diameter_m": d, "length_m": L, "steam_pressure_Pa": p,
                "insulation_thickness_m": t, "insulation_conductivity_W_m_K": k}.items() if v2 is not None})
    return {k: v for k, v in out.items() if v}
