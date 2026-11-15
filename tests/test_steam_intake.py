"""Steam diagnostic, all in one (bloques M20b and M21): distribution losses priced with the plant's steam, the Excel
workbook the plant fills, the one-command run and the HTML report. No network. The plant is invented (MOCKS).
"""
import copy
import json
import os

import pytest

from packages.dwsim_client import tools_steam as TS
from packages.energy import steam_audit as SA
from packages.energy import steam_cli as SC
from packages.energy import steam_distribution as SDI
from packages.energy import steam_html as SH
from packages.energy import steam_workbook as SW
from tests.test_steam_diagnostic import PLANT

DIST = {"leaks": [{"count": 4, "orifice_diameter_m": 0.003, "steam_pressure_Pa": 9.0e5}],
        "failed_traps": [{"count": 3, "orifice_diameter_m": 0.004, "steam_pressure_Pa": 9.0e5, "description": "línea 2"}],
        "bare_pipes": [{"outer_diameter_m": 0.1143, "length_m": 30.0, "steam_pressure_Pa": 9.0e5, "insulation_thickness_m": 0.05,
                        "insulation_conductivity_W_m_K": 0.045},
                       {"outer_diameter_m": 0.0603, "length_m": 12.0, "steam_pressure_Pa": 9.0e5}],
        "ambient_T_K": 298.15, "wind_speed_m_s": 0.0}


@pytest.fixture(scope="module")
def audited() -> dict:
    return TS.steam_system_diagnostic(**copy.deepcopy(PLANT), distribution=copy.deepcopy(DIST))


# ---------------------------------------------------------------------------------------------- distribution losses
def test_distribution_losses_are_the_engine_times_the_baseline_steam(audited):
    json.dumps(audited)
    d, base, hours = audited["distribution"], audited["baseline"], PLANT["operation"]["hours_per_year"]
    P_site = PLANT["combustion"]["site_pressure_Pa"]
    # leaks: the orifice engine with the published coefficient, to the site pressure, priced as steam leaving the system
    one = SDI.steam_leak_loss(9.0e5, 0.003, SA.MODEL_CONSTANTS["leak_discharge_coefficient"]["value"], P_site)
    assert d["leaks"]["steam_kg_h"] == pytest.approx(4 * one["mass_flow_kg_s"] * 3600.0, rel=1e-9)
    assert d["leaks"]["cost_per_year"] == pytest.approx(d["leaks"]["steam_kg_h"] * hours / 1000.0 * base["steam"]["cost_per_tonne"])
    assert d["leaks"]["co2_tonne_per_year"] == pytest.approx(
        d["leaks"]["steam_kg_h"] * hours / 1000.0 * base["emissions"]["co2_kg_per_tonne_steam"] / 1000.0)
    trap = SDI.steam_leak_loss(9.0e5, 0.004, SA.MODEL_CONSTANTS["trap_discharge_coefficient"]["value"], P_site)
    assert d["failed_traps"]["steam_kg_h"] == pytest.approx(3 * trap["mass_flow_kg_s"] * 3600.0, rel=1e-9)
    # bare pipe: heat priced with fuel only (the condensate stays in the system)
    pipes = d["bare_pipes"]
    assert pipes["cost_per_year"] == pytest.approx(pipes["heat_loss_W"] * hours * 3600.0 / 1e9
                                                   * base["steam"]["breakdown_per_GJ_delivered"]["fuel"])
    first, second = pipes["items"]
    assert first["insulation"]["saving_fraction"] > 0.85 and first["insulation"]["jacket_T_K"] < first["surface_T_K"]
    assert second["insulation"] is None and "falta el espesor" in second["insulation_note"]
    assert pipes["insulation_saving_cost_per_year"] == pytest.approx(first["insulation"]["saving_cost_per_year"])
    total = d["total"]
    assert total["cost_per_year"] == pytest.approx(d["leaks"]["cost_per_year"] + d["failed_traps"]["cost_per_year"]
                                                   + pipes["cost_per_year"])
    assert total["max_investment_for_payback"]["2"] == pytest.approx(2 * total["cost_per_year"])
    assert 0 < total["share_of_delivered_steam"] < SA.LOSS_SHARE_FLAG and d["evidence"]["verified"] is True
    # the four model constants are named with their source, never silent
    assert set(d["model_constants"]) == set(SA.MODEL_CONSTANTS)
    assert all(c["source"] for c in d["model_constants"].values())
    assert "DOE/GO-102012-3401" in SA.MODEL_CONSTANTS["trap_discharge_coefficient"]["source"]
    # the boiler study is untouched by the walk-through
    plain = TS.steam_system_diagnostic(**copy.deepcopy(PLANT))
    assert plain["distribution"] is None and plain["baseline"]["steam"]["cost_per_tonne"] == base["steam"]["cost_per_tonne"]
    assert any(m["parameter"] == "distribution" for m in plain["missing_data"])


def test_distribution_never_guesses_plant_data():
    def run(**change):
        return TS.steam_system_diagnostic(**copy.deepcopy(PLANT), distribution={**copy.deepcopy(DIST), **change})

    with pytest.raises(ValueError, match="ambient_T_K"):
        run(ambient_T_K=None)
    with pytest.raises(ValueError, match="orifice_diameter_m"):
        run(leaks=[{"count": 2, "steam_pressure_Pa": 9e5}])
    with pytest.raises(ValueError, match="número entero"):
        run(leaks=[{"count": 2.5, "orifice_diameter_m": 0.003, "steam_pressure_Pa": 9e5}])
    with pytest.raises(ValueError, match="espesor y la conductividad"):
        run(bare_pipes=[{"outer_diameter_m": 0.1, "length_m": 5.0, "steam_pressure_Pa": 9e5, "insulation_thickness_m": 0.05}])
    with pytest.raises(ValueError, match="claves desconocidas"):
        run(valves=[1])
    with pytest.raises(ValueError, match="al menos una fila"):
        TS.steam_system_diagnostic(**copy.deepcopy(PLANT), distribution={"ambient_T_K": 298.15})
    # a user constant replaces the published one and is labelled as such
    own = run(leak_discharge_coefficient=0.8)
    assert own["distribution"]["model_constants"]["leak_discharge_coefficient"]["origin"] == "dato del usuario"
    base = run()
    assert own["distribution"]["leaks"]["steam_kg_h"] == pytest.approx(base["distribution"]["leaks"]["steam_kg_h"] * 0.8 / 0.6)
    # a pipe diameter typed as the orifice diameter is flagged, not corrected
    huge = run(leaks=[{"count": 6, "orifice_diameter_m": 0.05, "steam_pressure_Pa": 9e5}])
    assert any(w.startswith("REVISAR DATO") and "orificio" in w for w in huge["warnings"])
    assert sum("orificio" in w for w in huge["warnings"]) == 1                             # said once, and in the data review
    assert [f["id"] for f in huge["data_review"]["flags"]] == ["distribution_share"]
    # the engine's note about totals does not apply: each insulated stretch has its own saving
    assert not any("None" in w for w in base["warnings"] + base["distribution"]["warnings"])
    # without hours: physical figures only
    kw = copy.deepcopy(PLANT)
    kw["operation"] = {}
    hourly = TS.steam_system_diagnostic(**kw, distribution=copy.deepcopy(DIST))["distribution"]
    assert hourly["leaks"]["steam_kg_h"] > 0 and hourly["total"]["cost_per_year"] is None


def test_report_has_the_distribution_section(audited):
    rep = TS.steam_diagnostic_report(inputs={**copy.deepcopy(PLANT), "distribution": copy.deepcopy(DIST)})
    md = rep["markdown"]
    assert md.index("## 4.") < md.index("## 4b. Pérdidas en la distribución") < md.index("## 5.")
    for needle in ("Fuga de vapor", "Trampa fallada abierta: línea 2", "Tubería sin aislar", "Con el aislamiento propuesto",
                   "Constantes del modelo que no se le piden a la planta", "DOE/GO-102012-3401"):
        assert needle in md, needle
    assert rep["verified"] is True


# -------------------------------------------------------------------------------------------------------- workbook
def _example(tmp_path, **changes) -> str:
    data = {**copy.deepcopy(SC.EXAMPLE), **changes}
    return SW.build_template(str(tmp_path / "datos.xlsx"), {k: v for k, v in data.items() if v is not None})


def test_workbook_round_trip_converts_plant_units(tmp_path):
    read = SW.read_workbook(_example(tmp_path))
    x = read["inputs"]
    p_atm = SW.standard_atmosphere_Pa(540.0)
    assert 9.4e4 < p_atm < 9.6e4 and x["combustion"]["site_altitude_m"] == 540.0 and "site_pressure_Pa" not in x["combustion"]
    assert read["notes"] == [] and read["plant_label"].startswith("Planta de ejemplo")
    assert ["Presión de operación", 8, "kg/cm² man."] in read["as_entered"]
    assert x["boiler"]["auxiliary_power_kWh_per_tonne_generated"] == 6.0
    assert x["boiler"]["steam_pressure_Pa"] == pytest.approx(8 * SW.KGF_CM2 + p_atm)          # gauge -> absolute
    assert x["boiler"]["rated_steam_capacity_kg_s"] == pytest.approx(600 * 15.65 / 3600.0)    # boiler horsepower
    assert x["combustion"]["stack_T_K"] == pytest.approx(493.15) and x["combustion"]["flue_O2"] == {"mole_fraction": 0.07, "basis": "dry"}
    assert x["combustion"]["setting_loss_unknown"] is True and "setting_loss_fraction" not in x["combustion"]
    assert x["fuel"] == {"type": "natural_gas_unknown", "T_K": 298.15}
    bill = x["fuel_cost"]["bill"]
    assert bill["energy_GJ"] == 9500.0 and bill["volume_m3"] == 248000.0 and bill["volume_reference"]["P_Pa"] == pytest.approx(SW.KGF_CM2)
    assert x["operation"] == {"hours_per_year": 6000.0, "steam_generated_kg_s": pytest.approx(6000.0 / 3600.0)}
    assert x["measures"]["reduce_excess_air"] == {"use_published_target": True}
    assert x["measures"]["recover_blowdown_heat"]["flash_pressure_Pa"] == pytest.approx(0.3 * SW.KGF_CM2 + p_atm)
    dist = x["distribution"]
    assert dist["leaks"][0]["count"] == 4 and dist["leaks"][0]["orifice_diameter_m"] == pytest.approx(0.003)
    assert dist["failed_traps"][0]["description"] == "línea de proceso 2" and dist["wind_speed_m_s"] == 0.0
    assert dist["bare_pipes"][0]["insulation_conductivity_W_m_K"] == 0.045 and "insulation_thickness_m" not in dist["bare_pipes"][1]
    # every field of the template has a row, a label and (numbers) a unit that the reader knows
    assert len(SW.FIELDS) == sum(len(fs) for _, fs in SW.SECTIONS) == 88
    out = TS.steam_system_diagnostic(**x)
    assert out["evidence"]["verified"] is True and out["distribution"]["evidence"]["verified"] is True
    # the atmosphere computed from the altitude is labelled as calculated, never as a plant datum
    assert out["baseline"]["site_pressure"] == {"Pa": pytest.approx(p_atm), "from_altitude_m": 540.0}
    assert not any("Presión atmosférica" in p["item"] for p in out["provenance"]["plant_data"])
    assert any(p["item"] == "Altitud del sitio" for p in out["provenance"]["plant_data"])
    assert any("Presión atmosférica" in c["figure"] and "atmósfera estándar" in c["tool"] for c in out["provenance"]["calculated"])
    # the plant's electricity figure is per tonne GENERATED; the engine carries it per tonne delivered
    g = out["baseline"]["steam"]["mass_per_kg_delivered"]["generated"]
    assert g > 1.0 and out["baseline"]["steam"]["power_kWh_per_tonne"] == pytest.approx(6.0 * g)
    assert out["baseline"]["annual"]["electricity_kWh"] == pytest.approx(6.0 * out["baseline"]["annual"]["steam_generated_tonne"])
    assert out["baseline"]["load_fraction"] == pytest.approx((6000 / 3600) / (600 * 15.65 / 3600))
    assert {v["id"] for v in out["assumed_values"]} == {"fuel_composition", "setting_loss", "target_reduce_excess_air",
                                                       "target_reduce_stack_temperature"}


def test_workbook_other_units_give_the_same_inputs(tmp_path):
    a = SW.read_workbook(_example(tmp_path))["inputs"]
    b = SW.read_workbook(_example(
        tmp_path, altitude=(540 / 0.3048, "ft"), stack_T=(428.0, "°F"), steam_pressure=(8 * SW.KGF_CM2 / SW.PSI, "psig"),
        steam_production=(6000 / 0.45359237, "lb/h"), fuel_bill_energy=(9500 / 1.055056, "MMBtu"),
        fuel_bill_vol_P=(98.0665, "kPa abs."), makeup_T=(293.15, "K")))["inputs"]
    assert b["combustion"]["stack_T_K"] == pytest.approx(a["combustion"]["stack_T_K"])
    assert b["boiler"]["steam_pressure_Pa"] == pytest.approx(a["boiler"]["steam_pressure_Pa"])
    assert b["operation"]["steam_generated_kg_s"] == pytest.approx(a["operation"]["steam_generated_kg_s"])
    assert b["fuel_cost"]["bill"]["energy_GJ"] == pytest.approx(9500.0)
    assert b["fuel_cost"]["bill"]["volume_reference"]["P_Pa"] == pytest.approx(a["fuel_cost"]["bill"]["volume_reference"]["P_Pa"])
    assert b["water"]["makeup_water_T_K"] == pytest.approx(293.15)
    # a known composition and a barometer reading replace the two assumptions
    c = SW.read_workbook(_example(tmp_path, comp_CH4=(95, "%"), comp_C2H6=(3, "%"), comp_N2=(2, "%"), p_atm=(95.0, "kPa"),
                                  setting_loss=(1.5, "%"), setting_loss_basis=("Superior (PCS)", None),
                                  setting_loss_unknown=None))
    assert c["inputs"]["fuel"]["composition"] == pytest.approx({"CH4": 0.95, "C2H6": 0.03, "N2": 0.02})
    assert c["inputs"]["combustion"]["site_pressure_Pa"] == pytest.approx(95000.0) and c["notes"] == []
    assert "site_altitude_m" not in c["inputs"]["combustion"]
    assert c["inputs"]["combustion"]["setting_loss_fraction"] == pytest.approx(0.015)


def test_workbook_lists_every_capture_error_with_its_cell(tmp_path):
    with pytest.raises(ValueError) as e:
        SW.read_workbook(_example(tmp_path, stack_T=("doscientos veinte", "°C"), steam_pressure=(8, "kilos"),
                                  setting_loss=(1.5, "%"), fuel_price=(150, "por GJ"), comp_CH4=(80, "%"),
                                  o2_basis=("no sé", None)))
    msg = str(e.value)
    for needle in ("no es un número", "elige la unidad de la lista", "«No la conozco»", "diste precio y también recibo",
                   "suma 80.00 %", "elige una opción de la lista", "Datos!B"):
        assert needle in msg, needle
    assert msg.count("\n- ") >= 6                                                         # all at once, not the first only
    # a gauge pressure without altitude or barometer cannot be made absolute
    with pytest.raises(ValueError, match="manométrica y falta la altitud"):
        SW.read_workbook(_example(tmp_path, altitude=None))
    # LP gas without composition is not fitted to a heating value
    with pytest.raises(ValueError, match="Solo el gas natural"):
        SW.read_workbook(_example(tmp_path, fuel_name=("Gas LP", None)))
    with pytest.raises(ValueError, match="no tiene la hoja"):
        from openpyxl import Workbook
        wb = Workbook()
        wb.save(str(tmp_path / "otro.xlsx"))
        SW.read_workbook(str(tmp_path / "otro.xlsx"))


def test_typical_capture_slips_are_flagged_not_fixed(tmp_path):
    # a percentage typed as a fraction: the number is used as typed, and the note says what to check
    slip = _example(tmp_path, blowdown=(0.08, "%"), m3_blowdown_target=None)
    read = SW.read_workbook(slip)
    assert read["inputs"]["boiler"]["blowdown_fraction"] == pytest.approx(0.0008)
    assert any(n.startswith("REVISAR DATO") and "8 %" in n and "Purga" in n for n in read["notes"])
    done = SC.run_workbook(slip, str(tmp_path / "slip"), pdf=False)                       # Markdown, HTML and tile agree
    md, page = open(done["files"]["markdown"]).read(), open(done["files"]["html"]).read()
    assert len(done["data_flags"]) == 1 and "Esa fila va en porcentaje" in md and "Esa fila va en porcentaje" in page
    assert "1 dato(s) por confirmar" in page and "ningún dato por confirmar" not in page
    assert not any(n.startswith("REVISAR DATO") for n in SW.read_workbook(_example(tmp_path))["notes"])
    # a stack temperature in °F typed with the °C unit
    hot = TS.steam_system_diagnostic(**SW.read_workbook(_example(tmp_path, stack_T=(428, "°C"), m2_published=None))["inputs"])
    assert "stack_far_above_saturation" in [f["id"] for f in hot["data_review"]["flags"]]


def test_empty_template_is_refused_by_the_engine_not_by_the_reader(tmp_path):
    book = SW.build_template(str(tmp_path / "vacia.xlsx"))
    read = SW.read_workbook(book)                                                          # empty cells are not capture errors
    assert read["inputs"] == {} and len(read["missing_essentials"]) == 16
    with pytest.raises(ValueError) as e:
        SC.run_workbook(book, str(tmp_path / "out"))
    msg = str(e.value)                                                                     # named as the plant sees them
    for needle in ("datos indispensables", "«Moneda de todos los importes»", "«Temperatura de los gases en la chimenea»",
                   "o marcar Sí en «No la conozco»", "Datos!B"):
        assert needle in msg, needle
    assert SW.read_workbook(_example(tmp_path))["missing_essentials"] == []
    # the feed-water temperature is not asked of a tank without heater, nor the condensate temperature with no return
    lean = SW.read_workbook(_example(tmp_path, fw_T=None, cond_T=None, cond_return=(0, "%"),
                                     fw_mode=("Tanque sin calentador (solo mezcla de condensado y agua de repuesto)", None)))
    assert lean["missing_essentials"] == []


# ------------------------------------------------------------------------------------------------- one command, HTML
def test_one_command_writes_the_deliverable(tmp_path, capsys):
    assert SC.main(["ejemplo", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Reporte listo" in printed and "Valores supuestos" in printed and "Balances comprobados: sí" in printed
    for name in ("datos_planta_ejemplo.xlsx", "reporte.html", "reporte.md", "resultado.json"):
        assert (tmp_path / name).stat().st_size > 1000, name
    page = (tmp_path / "reporte.html").read_text()
    md = (tmp_path / "reporte.md").read_text()
    result = json.loads((tmp_path / "resultado.json").read_text())
    d = result["diagnostic"]
    assert "CASO ILUSTRATIVO" in page and "INVENTADOS" in md
    # self-contained page: no external file, the three charts, the tiles, and the same headline numbers as the engine
    assert "<script" not in page and "http://" not in page and 'src="' not in page
    assert page.count("<svg") == 3 and page.count('class="tile"') >= 5
    assert f"{d['baseline']['annual']['cost']['total']:,.0f} MXN" in page
    assert f"{100 * d['baseline']['boiler']['efficiency']:.1f} %" in page
    assert f"{d['distribution']['total']['cost_per_year']:,.0f} MXN" in page
    assert "se calculó de la altitud" in page and "se calculó de la altitud" in md
    assert "## 9. Datos tal como los capturó la planta" in md and "| Presión de operación | 8 | kg/cm² man. |" in md
    assert "«Por tonelada» siempre es por tonelada de vapor entregado" in md
    # no internal parameter name reaches the client outside the provenance appendix (section 6)
    import re
    client = md[:md.index("## 6.")] + md[md.index("## 7."):]
    leaked = sorted(set(re.findall(r"[a-z]+_[A-Za-z_]+", client)) - {"steam_survey_guide", "boiler_efficiency"})
    assert leaked == [], leaked
    assert page.count("<table>") == sum(line.startswith("|---") for line in md.splitlines()) > 8   # every table became HTML
    assert "**" not in page.replace("<strong>", "") and "|---" not in page
    # the PDF is optional: with a Chrome it exists, without one the HTML is still the deliverable
    if SH.find_chrome():
        assert (tmp_path / "reporte.pdf").stat().st_size > 20000
    # capture errors come back as exit code 2 with the message on stderr
    bad = _example(tmp_path, stack_T=("x", "°C"))
    assert SC.main(["diagnostico", bad, "--salida", str(tmp_path / "bad"), "--sin-pdf"]) == 2
    assert "no es un número" in capsys.readouterr().err


def test_tools_expose_the_workbook_flow(tmp_path):
    made = TS.steam_workbook_template(str(tmp_path / "plantilla.xlsx"))
    assert os.path.exists(made["path"]) and made["sheets"] == ["Instrucciones", "Datos", "Fugas y trampas", "Tubería sin aislar"]
    book = _example(tmp_path)
    out = TS.steam_diagnostic_from_workbook(book, str(tmp_path / "salida"))
    json.dumps(out)
    assert out["verified"] is True and os.path.exists(out["files"]["html"]) and out["headline"]["cost_per_tonne"] > 0
    assert len(out["assumed_values"]) == 4 and out["data_flags"] == [] and out["not_evaluated"] == []
    from packages.agent import agent as AG
    from packages.dwsim_client import tools as T
    names = {f.__name__ for f in T.TOOLS}
    assert {"steam_workbook_template", "steam_diagnostic_from_workbook"} <= names
    assert "steam_diagnostic_from_workbook" in AG.SYSTEM_PROMPT if hasattr(AG, "SYSTEM_PROMPT") else True


def test_markdown_subset_and_bar_chart():
    blocks = SH.markdown_to_html("# T\n\nIntro **fuerte** <x>\n\n## Uno\n\n| a | b |\n|---|---|\n| 1 | **2** |\n\n- uno\n- dos\n\n> aviso\n")
    assert [t for t, _ in blocks] == ["", "Uno"]
    assert "<strong>fuerte</strong> &lt;x&gt;" in blocks[0][1]
    assert "<th>a</th>" in blocks[1][1] and "<td><strong>2</strong></td>" in blocks[1][1] and "<li>dos</li>" in blocks[1][1]
    assert "<blockquote><p>aviso</p></blockquote>" in blocks[1][1]
    svg = SH.bar_chart([("A", 10.0, "10"), ("B", 5.0, "5"), ("cero", 0.0, "0")], "Título", "sub")
    assert svg.count('class="bar"') == 2 and "<title>A: 10</title>" in svg and "cero" not in svg
    assert SH.bar_chart([], "x", "y") == ""
