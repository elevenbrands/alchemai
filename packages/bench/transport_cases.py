"""Bench v3 — transport phenomena (bloque 18): pipe ΔP with elevation, pipe sizing by velocity, shell-and-tube area.

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
