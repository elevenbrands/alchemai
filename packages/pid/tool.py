"""Agent / MCP tool `pid_from_hazop` (bloque M10): minimal P&ID as DEXPI P&ID from a flowsheet and its HAZOP."""
from __future__ import annotations

from pathlib import Path

from packages.pid import dexpi_pid as DP
from packages.pid.rules import ALARM_SEVERITY, INLINE, RULES, TRIP_SEVERITY, VAR_ES, PidSpec, build_pid

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "exports"
CATEGORY_ES = {"lazo": "lazo de control", "alarma": "alarma", "disparo": "disparo (SIS)", "indicador": "indicador",
               "aislamiento": "válvula de aislamiento", "check": "válvula check", "valvula_control": "válvula de control",
               "valvula_manual": "válvula manual", "alivio": "válvula de alivio"}


def tables(spec: PidSpec) -> dict:
    """Spanish tables: instruments, loops and piping items, each with its rule, source and HAZOP rows."""
    instrumentos = [{"tag": i.tag, "tipo": CATEGORY_ES[i.category], "transmisor": i.sensor, "equipo": i.unit,
                     "medición": f"línea {i.sensing_line}" if i.sensing_line else f"en {i.unit}",
                     "elemento_final": i.final_element, "consigna_o_límite": i.setpoint, "descripción": i.detail,
                     "acción": i.action, "regla": i.rule, "fuente": RULES[i.rule][1], "hazop": i.hazop, "origen": i.origin}
                    for i in spec.instruments]
    lazos = []
    for no in sorted({i.loop for i in spec.instruments}):
        g = [i for i in spec.instruments if i.loop == no]
        main = g[0]
        lazos.append({"lazo": no, "variable": VAR_ES[main.variable], "equipo": main.unit,
                      "tipo": "SIS" if main.category == "disparo" else "BPCS",
                      "funciones": [i.tag for i in g], "transmisor": main.sensor, "elemento_final": main.final_element,
                      "cascada_a": main.cascade_to, "descripción": main.detail})
    elementos = [{"tag": it.tag, "tipo": CATEGORY_ES[it.category], "línea": it.line or f"boquilla en {it.unit}",
                  "equipo": it.unit, "válvula_del_flowsheet": it.dwsim, "descripción": it.detail, "dato_pendiente": it.setpoint,
                  "regla": it.rule, "fuente": RULES[it.rule][1], "hazop": it.hazop, "origen": it.origin} for it in spec.items]
    return {"instrumentos": instrumentos, "lazos": lazos, "elementos_de_tuberia": elementos}


def markdown(spec: PidSpec) -> str:
    t = tables(spec)
    out = ["| Lazo | Variable | Equipo | Funciones | Elemento final | Cascada | Descripción |", "|---|---|---|---|---|---|---|"]
    out += [f"| {l['lazo']} | {l['variable']} | {l['equipo']} | {', '.join(l['funciones'])} | {l['elemento_final']} | "
            f"{l['cascada_a'] or '—'} | {l['descripción']} |" for l in t["lazos"]]
    out += ["", "| Tag | Tipo | Línea | Regla | Filas HAZOP | Descripción |", "|---|---|---|---|---|---|"]
    out += [f"| {e['tag']} | {e['tipo']} | {e['línea']} | {e['regla']} | {len(e['hazop'])} | {e['descripción']} |"
            for e in t["elementos_de_tuberia"]]
    return "\n".join(out)


def pid_from_graph(g, out_dir: str | Path | None, stem: str, alarm_severity: int = ALARM_SEVERITY,
                   trip_severity: int = TRIP_SEVERITY, fetch_xsd: bool = True) -> dict:
    spec = build_pid(g, alarm_severity=alarm_severity, trip_severity=trip_severity)
    model = DP.to_dexpi_pid(spec)
    out = {"name": spec.name, **tables(spec), "no_instrumentadas": spec.not_instrumented, "indeterminado": spec.undetermined,
           "assumptions": spec.assumptions, "reglas": {k: {"regla": v[0], "fuente": v[1]} for k, v in RULES.items()},
           "conteo": {"equipos": sum(1 for k in spec.units.values() if k not in INLINE), "lineas": len(spec.lines),
                      "lazos": len({i.loop for i in spec.instruments}), "instrumentos": len(spec.instruments),
                      "elementos_de_tuberia": len(spec.items)},
           "propuestas_llm": [], "markdown": markdown(spec)}
    ev = {"semantic_issues": DP.validate_semantic(model)}
    if out_dir is not None:
        files = DP.save(model, out_dir, stem)
        out["files"] = files
        d = DP.digest(model)
        ev["roundtrip_xml"] = DP.digest(DP.load(files["dexpi_pid_xml"])) == d
        ev["roundtrip_json"] = DP.digest(DP.load(files["dexpi_pid_json"])) == d
        ev["matches_spec"] = DP.model_matches_spec(DP.load(files["dexpi_pid_xml"]), spec)
        ev["xsd"] = DP.validate_xsd(files["dexpi_pid_xml"], fetch=fetch_xsd)
        ok = (ev["roundtrip_xml"] and ev["roundtrip_json"] and all(v for k, v in ev["matches_spec"].items() if k != "diff")
              and not ev["semantic_issues"])
        out["verified"] = bool(ok and ev["xsd"]["status"] == "valido")
        if ok and ev["xsd"]["status"] == "pendiente":
            out["verified"] = False
            ev["note"] = "ida y vuelta correcta; validación XSD pendiente (sin red)"
    else:
        out["verified"] = False
    out["evidence"] = ev
    return out


def pid_from_hazop(flowsheet_id: str | None = None, dexpi_path: str | None = None, out_dir: str | None = None,
                   alarm_severity: int = ALARM_SEVERITY, trip_severity: int = TRIP_SEVERITY) -> dict:
    """Minimal P&ID of a SOLVED flowsheet (flowsheet_id) or of a DEXPI Process file (dexpi_path), written as a
    DEXPI 2.0.1 P&ID (XML + JSON) with pyDEXPI. Deterministic rules add level loops on liquid inventories, pressure
    loops on gas holdups, flow loops on pump/compressor discharges, temperature loops on heaters, coolers and
    reactors, the LV scheme on columns, isolation valves at battery limits and rotating equipment; the HAZOP adds
    relief valves, check valves, alarms (severity >= alarm_severity) and trips (>= trip_severity) only where its
    rows ask for them. Every element carries an ISA-5.1 tag and its trace (rule, source, HAZOP rows). Setpoints,
    alarm limits and PSV set pressures stay blank ("por definir"). The file is re-read and compared (round trip),
    checked with pyDEXPI's rules and validated against the official DEXPI XSD."""
    if bool(flowsheet_id) == bool(dexpi_path):
        raise ValueError("da flowsheet_id (flowsheet resuelto) o dexpi_path (DEXPI Process), uno de los dos")
    if flowsheet_id:
        from packages.dwsim_client import tools as T
        from packages.hazop.graph import build_graph
        g = build_graph(T._s(flowsheet_id)["fs"], flowsheet_id)
        stem, missing = flowsheet_id, []
    else:
        from packages.hazop.dexpi_graph import graph_from_dexpi, load_dexpi
        p = Path(dexpi_path)
        g, missing = graph_from_dexpi(load_dexpi(p), p.stem)
        stem = p.stem
    out = pid_from_graph(g, Path(out_dir) if out_dir else DEFAULT_OUT / stem / "pid", f"{stem}_pid", alarm_severity, trip_severity)
    out["missing"] = missing
    return out
