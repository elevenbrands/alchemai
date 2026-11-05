"""Bench v3 — transport phenomena (bloque 18: pipe ΔP, line size, shell-and-tube area) and equipment cost (bloque 19:
exchanger and pump purchase cost by Seider et al. via IDAES, chained to the sizing).

Oracles: DWSIM 10.2.12 for the stream properties, the pipe ΔP (Pipe Segment, Beggs-Brill) and the duty; the area
uses LMTD·F (Fakheri) from `packages.transport.hydraulics`, cross-checked in notebook 12 against DWSIM's
HeatExchanger (CalcBothTemp_UA recovers A within 0.3 %).
Regenerate only these cards: PYTHONPATH=. uv run python packages/bench/transport_cases.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.bench.design_cases import DesignCase  # noqa: E402
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_transport import pipe_hydraulics, size_pipe  # noqa: E402
from packages.transport.hydraulics import hx_area  # noqa: E402

NPS3_SCH40_ID = 0.07793


def _oracle_dp():
    fid = T.new_flowsheet("v3_dp", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=293.15, pressure_Pa=5e5, mass_flow_kg_s=10.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "out")
    T.add_unit(fid, "pipe", "L-1", {"length_m": 250.0, "inner_diameter_m": NPS3_SCH40_ID, "elevation_change_m": 15.0, "increments": 20})
    T.connect(fid, "L-1", feeds=["in"], products=["out"])
    r = T.solve(fid)
    assert r["solved"], r
    u = T.read_unit(fid, "L-1")
    ours = pipe_hydraulics(fid, "in", 250.0, NPS3_SCH40_ID, elevation_change_m=15.0)
    T.close_flowsheet(fid)
    return {"P_out_bar": r["streams"]["out"]["P_Pa"] / 1e5, "dP_friction_kPa": u["dP_friction_Pa"] / 1e3,
            "velocity_m_s": ours["velocity_m_s"], "_ours_dP_friction_kPa": ours["dP_friction_Pa"] / 1e3}


def _oracle_size():
    fid = T.new_flowsheet("v3_size", ["N-hexane"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=300.0, pressure_Pa=5e5, mass_flow_kg_s=30.0, mole_fractions={"N-hexane": 1.0})
    T.add_material_stream(fid, "out")
    T.add_unit(fid, "valve", "V-1", {"outlet_P_Pa": 4.9e5})          # any unit, so the stream is part of a solved flowsheet
    T.connect(fid, "V-1", feeds=["in"], products=["out"])
    assert T.solve(fid)["solved"]
    s = size_pipe(fid, "in", 2.0)
    T.close_flowsheet(fid)
    return {"NPS_in": s["NPS_in"], "velocity_m_s": s["velocity_m_s"], "dP_per_100m_kPa": s["dP_friction_per_100m_Pa"] / 1e3,
            "_inner_diameter_m": s["inner_diameter_m"]}


def _oracle_area():
    fid = T.new_flowsheet("v3_area", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=360.0, pressure_Pa=3e5, mass_flow_kg_s=8.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "out")
    T.add_unit(fid, "cooler", "E-1", {"outlet_T_K": 320.0})
    T.connect(fid, "E-1", feeds=["in"], products=["out"])
    assert T.solve(fid)["solved"]
    Q = abs(T.read_unit(fid, "E-1")["heat_removed_kW"])
    T.close_flowsheet(fid)
    a = hx_area(Q * 1e3, 360.0, 320.0, 300.0, 315.0, 850.0, shell_passes=1)
    return {"Q_kW": Q, "LMTD_K": a["LMTD_K"], "F": a["F"], "area_m2": a["area_m2"]}


def _oracle_cost_hx():
    from packages.costing.sslw import cost_heat_exchanger
    a = _oracle_area()
    c = cost_heat_exchanger(a["area_m2"], shell_pressure_Pa=3e5, hx_type="Utube", material="StainlessSteelStainlessSteel",
                            tube_length="TwelveFoot", year=2023)
    return {"area_m2": a["area_m2"], "purchase_cost_usd": c["purchase_cost_usd"]}


def _oracle_cost_pump():
    from packages.dwsim_client.tools_costing import cost_unit
    fid = T.new_flowsheet("v3_pump", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=298.15, pressure_Pa=1e5, mass_flow_kg_s=50.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "out"); T.add_energy_stream(fid, "w")
    T.add_unit(fid, "pump", "P-1", {"outlet_P_Pa": 6e5, "efficiency_pct": 75})
    T.connect(fid, "P-1", feeds=["in"], products=["out"], energy_feeds=["w"])
    assert T.solve(fid)["solved"]
    c = cost_unit(fid, "P-1", {"pump_type": "Centrifugal", "material": "CastIron", "pump_type_factor": 1.4, "motor_type": "Open"}, 2023)
    T.close_flowsheet(fid)
    return {"power_kW": c["size_from_simulation"]["power_W"] / 1e3, "purchase_cost_usd": c["purchase_cost_usd"]}



def _oracle_heat_recovery():
    from packages.costing.heat_recovery import optimize
    b = optimize()["best"]
    return {"area_m2": b["area_E1_m2"], "approach_K": b["approach_K"], "TAC_usd_y": b["TAC_usd_y"]}


TRANSPORT_CASES = [
    DesignCase(
        "transporte_dp_tuberia", "Caída de presión en una línea de agua que sube 15 m", "Pressure drop in a water line rising 15 m",
        "10 kg/s de agua a 20 °C y 5 bar fluyen por 250 m de tubería de acero comercial de 3 in nominal cédula 40 "
        "(diámetro interno 77.93 mm); la salida está 15 m más arriba que la entrada. ¿Presión de salida (bar), caída de presión "
        "por fricción (kPa) y velocidad (m/s)?",
        "10 kg/s of water at 20 °C and 5 bar flow through 250 m of 3 in nominal schedule 40 commercial steel pipe (inner diameter "
        "77.93 mm); the outlet is 15 m above the inlet. Outlet pressure (bar), frictional pressure drop (kPa) and velocity (m/s)?",
        _oracle_dp, {"P_out_bar": {"rel": 0.02}, "dP_friction_kPa": {"rel": 0.03}, "velocity_m_s": {"rel": 0.02}},
        {"P_out_bar": "bar", "dP_friction_kPa": "kPa", "velocity_m_s": "m/s"},
        lambda a: [("Darcy propio vs DWSIM < 1 %", abs(a["_ours_dP_friction_kPa"] / a["dP_friction_kPa"] - 1) < 0.01, a["_ours_dP_friction_kPa"])],
        ["transporte", "hidraulica"]),
    DesignCase(
        "transporte_diametro", "Diámetro de una línea de n-hexano por velocidad", "Line size for n-hexane by velocity",
        "30 kg/s de n-hexano líquido a 300 K y 5 bar (Peng-Robinson). ¿Cuál es el menor diámetro nominal de acero cédula 40, "
        "entre los tamaños comerciales (sin 1¼, 2½, 3½ ni 5 in), que mantiene la velocidad en 2 m/s o menos? Da el diámetro "
        "nominal (in), la velocidad (m/s) y la caída de presión por fricción por cada 100 m (kPa).",
        "30 kg/s of liquid n-hexane at 300 K and 5 bar (Peng-Robinson). What is the smallest schedule 40 steel nominal size, among "
        "common sizes (no 1¼, 2½, 3½ or 5 in), that keeps the velocity at or below 2 m/s? Give the nominal size (in), the velocity "
        "(m/s) and the frictional pressure drop per 100 m (kPa).",
        _oracle_size, {"NPS_in": {"abs": 0.0}, "velocity_m_s": {"rel": 0.03}, "dP_per_100m_kPa": {"rel": 0.05}},
        {"NPS_in": "in", "velocity_m_s": "m/s", "dP_per_100m_kPa": "kPa"},
        lambda a: [("velocidad ≤ 2 m/s", a["velocity_m_s"] <= 2.0, a["velocity_m_s"])],
        ["transporte", "hidraulica", "diseno"]),
    DesignCase(
        "transporte_area_intercambiador", "Área de un enfriador de coraza y tubos 1-2", "Area of a 1-2 shell-and-tube cooler",
        "8 kg/s de agua a 360 K y 3 bar se enfrían a 320 K con agua de enfriamiento que entra a 300 K y sale a 315 K, en un "
        "intercambiador de coraza y tubos con una coraza y dos pasos de tubo. U = 850 W/m²K. ¿Carga térmica (kW), LMTD en "
        "contracorriente (K), factor de corrección F y área (m²)?",
        "8 kg/s of water at 360 K and 3 bar are cooled to 320 K with cooling water entering at 300 K and leaving at 315 K, in a "
        "shell-and-tube exchanger with one shell and two tube passes. U = 850 W/m²K. Heat duty (kW), counter-current LMTD (K), "
        "correction factor F and area (m²)?",
        _oracle_area, {"Q_kW": {"rel": 0.02}, "LMTD_K": {"rel": 0.01}, "F": {"abs": 0.01}, "area_m2": {"rel": 0.03}},
        {"Q_kW": "kW", "LMTD_K": "K", "F": "-", "area_m2": "m2"},
        lambda a: [("F entre 0.75 y 1", 0.75 <= a["F"] <= 1.0, a["F"]), ("Q ≈ m·cp·ΔT (±2 %)", abs(a["Q_kW"] / (8 * 4.19 * 40) - 1) < 0.02, a["Q_kW"])],
        ["transporte", "calor", "diseno"]),
    DesignCase(
        "costo_intercambiador", "Área y costo de compra de un enfriador 1-2", "Area and purchase cost of a 1-2 cooler",
        "8 kg/s de agua a 360 K y 3 bar se enfrían a 320 K con agua de enfriamiento de 300 a 315 K en un intercambiador de "
        "coraza y tubos de tubo en U, una coraza y dos pasos de tubo, acero inoxidable en ambos lados, tubos de 12 ft, "
        "U = 850 W/m²K, presión de diseño de 3 bar. ¿Área (m²) y costo de compra en USD de 2023 según Seider et al.?",
        "8 kg/s of water at 360 K and 3 bar are cooled to 320 K with cooling water from 300 to 315 K in a U-tube shell-and-tube "
        "exchanger, one shell and two tube passes, stainless steel on both sides, 12 ft tubes, U = 850 W/m²K, 3 bar design "
        "pressure. Area (m²) and purchase cost in 2023 USD per Seider et al.?",
        _oracle_cost_hx, {"area_m2": {"rel": 0.03}, "purchase_cost_usd": {"rel": 0.05}}, {"area_m2": "m2", "purchase_cost_usd": "USD"},
        lambda a: [("costo > 0", a["purchase_cost_usd"] > 0, a["purchase_cost_usd"])], ["costo", "calor", "diseno"]),
    DesignCase(
        "costo_bomba", "Potencia y costo de compra de una bomba centrífuga", "Power and purchase cost of a centrifugal pump",
        "Una bomba centrífuga lleva 50 kg/s de agua a 25 °C de 1 a 6 bar con eficiencia de 75 %. Carcasa de hierro fundido, "
        "motor abierto, factor de tipo de bomba 1.4 (tabla de Seider). ¿Potencia (kW) y costo de compra en USD de 2023 según "
        "Seider et al.?",
        "A centrifugal pump moves 50 kg/s of water at 25 °C from 1 to 6 bar at 75 % efficiency. Cast-iron casing, open motor, "
        "pump type factor 1.4 (Seider's table). Power (kW) and purchase cost in 2023 USD per Seider et al.?",
        _oracle_cost_pump, {"power_kW": {"rel": 0.02}, "purchase_cost_usd": {"rel": 0.05}}, {"power_kW": "kW", "purchase_cost_usd": "USD"},
        lambda a: [("potencia ≈ ΔP·Q/η", abs(a["power_kW"] / (5e5 * 50 / 997 / 0.75 / 1e3) - 1) < 0.01, a["power_kW"])], ["costo", "diseno"]),
    DesignCase(
        "optimo_recuperacion_calor", "Área óptima de un intercambiador alimentación-producto", "Optimal area of a feed/product exchanger",
        "Agua de alimentación, 10 kg/s a 300 K y 4 bar, debe llegar a 395 K; agua de producto, 10 kg/s a 400 K y 4 bar, debe salir "
        "a 305 K (IAPWS). Un intercambiador alimentación-producto E-1 a contracorriente (U = 850 W/m²K) recupera calor; un "
        "calentador con vapor que condensa a 450 K (U = 1100 W/m²K) completa la alimentación y un enfriador con agua de "
        "enfriamiento de 290 a 300 K (U = 850 W/m²K) completa el producto, ambos a contracorriente sin corrección F. Los tres son "
        "de coraza y tubos en U, acero al carbono en ambos lados, tubos de 12 ft, presión de diseño 4 bar; costo de compra de "
        "Seider et al. en USD de 2023 (si un área queda bajo el mínimo de la correlación, se cotiza al mínimo). Capital fijo = "
        "compra × 4.74 (Lang, planta de fluidos), anualizado al 10 % a 10 años. Vapor 8 USD/GJ, agua de enfriamiento 0.5 USD/GJ, "
        "8000 h/año. ¿Qué área de E-1 (m²) minimiza el costo total anual, con qué aproximación de temperatura (K) y cuál es ese "
        "costo total anual (USD/año)?",
        "Feed water, 10 kg/s at 300 K and 4 bar, must reach 395 K; product water, 10 kg/s at 400 K and 4 bar, must leave at 305 K "
        "(IAPWS). A counter-current feed/product exchanger E-1 (U = 850 W/m²K) recovers heat; a heater with steam condensing at "
        "450 K (U = 1100 W/m²K) finishes the feed and a cooler with cooling water from 290 to 300 K (U = 850 W/m²K) finishes the "
        "product, both counter-current with no F correction. All three are U-tube shell-and-tube, carbon steel on both sides, "
        "12 ft tubes, 4 bar design pressure; Seider et al. purchase cost in 2023 USD (an area below the correlation minimum is "
        "priced at the minimum). Fixed capital = purchase × 4.74 (Lang, fluid plant), annualized at 10 % over 10 years. Steam "
        "8 USD/GJ, cooling water 0.5 USD/GJ, 8000 h/y. Which E-1 area (m²) minimizes total annual cost, at what temperature "
        "approach (K), and what is that total annual cost (USD/y)?",
        _oracle_heat_recovery, {"area_m2": {"rel": 0.20}, "approach_K": {"abs": 1.5}, "TAC_usd_y": {"rel": 0.02}},
        {"area_m2": "m2", "approach_K": "K", "TAC_usd_y": "USD/y"},
        lambda a: [("aproximación entre 5 y 15 K", 5.0 < a["approach_K"] < 15.0, a["approach_K"])], ["costo", "optimizacion", "calor"]),
]


def cards() -> list[dict]:
    from packages.bench.build_cards import _card
    out = []
    for d in TRANSPORT_CASES:
        a = d.oracle()
        checks = d.checks(a)
        assert all(ok for _, ok, _ in checks), f"{d.id}: {checks}"
        out.append(_card(d.id, "design", (d.title_es, d.title_en), (d.statement_es, d.statement_en), d.tags,
                         {k: v for k, v in a.items() if k in d.tolerances}, d.units, d.tolerances, version="v3"))
    return out


if __name__ == "__main__":
    import yaml
    from packages.bench.build_cards import CARDS_DIR, SOURCE
    cs = cards()
    for c in cs:
        (CARDS_DIR / f"{c['id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=110))
        print(c["id"], {k: round(v["value"], 4) for k, v in c["expected"].items()})
    idx = yaml.safe_load((CARDS_DIR / "index.yaml").read_text())
    ids = {c["id"] for c in cs}
    idx["cards"] = [x for x in idx["cards"] if x["id"] not in ids] + [
        {"id": c["id"], "kind": c["kind"], "version": c["version"], "title_es": c["title"]["es"], "n_expected": len(c["expected"])} for c in cs]
    (CARDS_DIR / "index.yaml").write_text(yaml.safe_dump(idx | {"source": SOURCE}, sort_keys=False, allow_unicode=True))
    print(len(idx["cards"]), "cards in index")
