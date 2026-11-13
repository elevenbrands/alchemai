"""Agent tools for energy integration: pinch analysis on a stream table (bloque M5) and straight from a built DWSIM
flowsheet (bloque M5b: heaters, coolers and exchanger sides extracted, CP segmented from DWSIM flashes). Engines and
evidence live in `packages.energy.pinch` and `packages.energy.stream_extraction`."""
from __future__ import annotations

from packages.energy import pinch as P
from packages.energy import stream_extraction as X


def pinch_analysis(streams: list[dict], dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
                   current_hot_utility_W: float | None = None, current_cold_utility_W: float | None = None) -> dict:
    """Pinch analysis of a process stream table by the Problem Table Algorithm: minimum hot and cold utility (MER
    targets), pinch temperature, maximum heat recovery, problem table, heat cascade, grand composite curve (where to
    place utilities: low- vs high-pressure steam, cooling water vs refrigeration), hot and cold composite curves in
    real temperatures, minimum number of exchangers, and which streams cross the pinch. streams = [{'name',
    'T_supply_K', 'T_target_K', 'CP_W_K' (m·cp, assumed constant) or 'heat_load_W', optional 'dT_contribution_K'
    (default dT_min_K/2; use larger for gases/low h, smaller for boiling/condensing)}]; hot streams are those that
    cool (T_supply > T_target). A phase change at constant T goes as a 1 K span with heat_load_W = latent duty; a
    stream with strongly varying cp should be split into segments. dT_min_K is a design decision (typical 10-20 K
    liquids, 20-40 K gases, 3-5 K refrigeration): ask or state the one used, and give dT_min_sweep_K to show the
    trade-off. current_hot_utility_W / current_cold_utility_W (the plant's present consumption) give the saving
    potential = heat crossing the pinch. evidence: the shifted composite curves built independently must touch at the
    pinch without crossing, a smaller cold utility must make them cross, and the overall energy balance must close."""
    return P.pinch(streams, dT_min_K, dT_min_sweep_K, current_hot_utility_W, current_cold_utility_W)


_HEAT_KINDS = ("heater", "cooler", "heat_exchanger")
_OTHER_DUTY_KINDS = ("shortcut_column", "distillation_column", "absorption_column", "conversion_reactor",
                     "equilibrium_reactor", "gibbs_reactor", "cstr", "pfr")


def _pieces(s: dict, rep) -> tuple[list, list[str]]:
    """One X.Piece per heater/cooler and per exchanger side, from the solved report; plus skip notes."""
    units = {u.name: u for u in rep.units}
    pieces, skipped = [], []
    for name, kind in s["kinds"].items():
        if kind not in _HEAT_KINDS:
            continue
        c, u = s["conn"].get(name) or {}, units.get(name)
        feeds, prods = c.get("feeds") or [], c.get("products") or []
        n = 2 if kind == "heat_exchanger" else 1
        if u is None or len(feeds) < n or len(prods) < n:
            skipped.append(f"{name}: sin conectar o sin resolver")
            continue
        duty_W = abs(float(u.duty_kw or 0.0)) * 1e3
        sides = [("hot", feeds[0], prods[0]), ("cold", feeds[1], prods[1])] if n == 2 else [("", feeds[0], prods[0])]
        for side, a, b in sides:
            va, vb = rep.streams[a], rep.streams[b]
            p = X.Piece(unit=name, kind=kind, side=side, inlet=a, outlet=b, T_in=va.temperature_k, T_out=vb.temperature_k,
                        P_in=va.pressure_pa, P_out=vb.pressure_pa, H_in=va.enthalpy_kw * 1e3, H_out=vb.enthalpy_kw * 1e3,
                        duty_reported_W=duty_W)
            if p.Q <= 1e-6 * max(abs(p.H_in), 1.0):
                skipped.append(f"{name}{'(' + side + ')' if side else ''}: sin carga térmica")
                continue
            p._flows = dict(va.compounds_mol_s)
            pieces.append(p)
    return pieces, skipped


def _dwsim_flash(T, s: dict, pieces: list):
    """Enthalpy (W) of each piece's material at (T, P): one auxiliary flowsheet with the same compounds and property
    package, one free stream per request, solved in a single call and discarded."""
    def flash(reqs):
        if not reqs:
            return []
        fid = T.new_flowsheet("pinch_aux", s["compounds"], s["meta"]["property_package"])["flowsheet_id"]
        try:
            for k, (i, Tk, Pk) in enumerate(reqs):
                T.add_material_stream(fid, f"x{k}", temperature_K=Tk, pressure_Pa=Pk,
                                      compound_molar_flows_mol_s={c: f for c, f in pieces[i]._flows.items() if f > 0})
            r = T.solve(fid)
            if r["solver_errors"]:
                raise RuntimeError(f"flash auxiliar falló: {r['solver_errors'][:2]}")
            st = T.SESSIONS[fid]["report"].streams
            return [st[f"x{k}"].enthalpy_kw * 1e3 for k in range(len(reqs))]
        finally:
            T.close_flowsheet(fid)
    return flash


def pinch_from_flowsheet(flowsheet_id: str, dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
                         linearization_tol_pct: float = 1.0, dT_contribution_K: dict[str, float] | None = None,
                         exclude_units: list[str] | None = None) -> dict:
    """Pinch analysis straight from a built DWSIM flowsheet: re-solves it, takes every heater, cooler and heat
    exchanger side as a process stream (supply T = inlet of the first unit, target T = outlet of the last; a stream
    going exchanger → cooler is ONE stream), builds its H(T) curve by DWSIM flashes and splits it into constant-CP
    segments wherever cp varies or the phase changes (error ≤ linearization_tol_pct % of the stream duty), counts
    heaters as current hot utility and coolers as current cold utility, and runs pinch_analysis. dT_min_K is the
    user's design choice (ask; never assume it): give dT_min_sweep_K to show the trade-off. dT_contribution_K =
    {stream name from a previous call: K} overrides ΔTmin/2. Returns the extracted stream table (check it with the
    user), the per-unit balance (ΔH of the streams vs the duty DWSIM reports, tolerance BALANCE_REL_TOL), MER targets,
    pinch, saving vs current utilities, the same targets with one linear CP per stream (shows what segmentation
    changes), assumptions and evidence. Column condensers/reboilers and reactor duties are NOT extracted (listed)."""
    from packages.dwsim_client import tools as T          # lazy: keeps pinch_analysis free of DWSIM
    if linearization_tol_pct <= 0:
        raise ValueError("linearization_tol_pct must be > 0")
    s = T._s(flowsheet_id)
    sol = T.solve(flowsheet_id)
    if not sol["ok"]:
        raise ValueError("el flowsheet no está resuelto y verificado (solve ok=false); arréglalo antes del pinch")
    rep = s["report"]
    pieces, skipped = _pieces(s, rep)
    excl = set(exclude_units or [])
    pieces = [p for p in pieces if p.unit not in excl]
    X.sample(pieces, _dwsim_flash(T, s, pieces), linearization_tol_pct)
    tab = X.stream_table(pieces, linearization_tol_pct, dT_contribution_K)
    utilities = [{"unit": p.unit, "kind": p.kind, "utility": "frío" if p.hot else "caliente", "duty_W": p.Q,
                  "T_in_K": p.T_in, "T_out_K": p.T_out} for p in pieces if p.kind != "heat_exchanger"]
    recovered = {p.unit: p.Q for p in pieces if p.kind == "heat_exchanger" and p.side == "hot"}
    hot_now = sum(u["duty_W"] for u in utilities if u["utility"] == "caliente")
    cold_now = sum(u["duty_W"] for u in utilities if u["utility"] == "frío")
    if not any(r["type"] == "hot" for r in tab["rows"]) or not any(r["type"] == "cold" for r in tab["rows"]):
        raise ValueError("el flowsheet no tiene corrientes calientes y frías a la vez: no hay integración posible")
    res = P.pinch(tab["segmented"], dT_min_K, dT_min_sweep_K, hot_now, cold_now)
    lin = P.pinch(tab["linear"], dT_min_K)
    units = X.units_by_parent(res, tab["parent_of"])
    if units is not None:
        res["minimum_units"] = units
    crossing = sorted({tab["parent_of"].get(x["name"], x["name"]) for x in res["streams"] if x["crosses_pinch"]})
    others = [f"{n} ({k})" for n, k in s["kinds"].items() if k in _OTHER_DUTY_KINDS]
    hx_ok = all(abs(a.Q - b.Q) <= X.BALANCE_REL_TOL * max(a.duty_reported_W, 1.0)
                for a in pieces for b in pieces if a.unit == b.unit and a.side == "hot" and b.side == "cold")
    # global: Σ cold − Σ hot of the extracted streams must equal hot − cold current utility (exchangers cancel)
    Qc = sum(r["duty_W"] for r in tab["rows"] if r["type"] == "cold")
    Qh = sum(r["duty_W"] for r in tab["rows"] if r["type"] == "hot")
    glob = abs((Qc - Qh) - (hot_now - cold_now))
    glob_ok = glob <= X.BALANCE_REL_TOL * max(Qc, Qh, 1.0)
    aux_bad = [c["unit"] for c in tab["checks"] if c["aux_flash_endpoint_err_W"] is not None
               and c["aux_flash_endpoint_err_W"] > linearization_tol_pct / 100.0 * c["dH_streams_W"]]
    reasons = list(res["evidence"]["reasons"])
    for c in tab["checks"]:
        if not c["ok"]:
            reasons.append(f"{c['unit']}: ΔH de las corrientes {c['dH_streams_W']:.6g} W vs duty de DWSIM "
                           f"{c['duty_reported_W']:.6g} W (error {c['rel_err']:.2e} > {X.BALANCE_REL_TOL})")
        if not c["segments_close"]:
            reasons.append(f"{c['unit']}: los segmentos no suman la carga")
    if not hx_ok:
        reasons.append("un intercambiador no transfiere lo mismo en sus dos lados")
    if not glob_ok:
        reasons.append(f"balance global de corrientes extraídas vs servicios actuales no cierra ({glob:.3g} W)")
    warnings = list(res["warnings"])
    if skipped:
        warnings.append("omitidos: " + "; ".join(skipped))
    if excl:
        warnings.append("excluidos por el usuario: " + ", ".join(sorted(excl)))
    if others:
        warnings.append("con carga térmica pero NO extraídos (condensadores/rehervidores/reactores): " + ", ".join(others)
                        + "; el servicio actual y los objetivos no los incluyen")
    if aux_bad:
        warnings.append("el flash auxiliar no reproduce la entalpía del flowsheet en los extremos de: " + ", ".join(aux_bad)
                        + " (corriente saturada o especificada por fracción de vapor); los puntos interiores pueden "
                        "desviarse; revisar sus segmentos")
    res["warnings"] = warnings
    res["evidence"] = {**res["evidence"], "unit_balances_ok": all(c["ok"] for c in tab["checks"]),
                       "exchangers_balanced": hx_ok, "global_balance_closed": glob_ok, "global_balance_err_W": glob,
                       "balance_rel_tol": X.BALANCE_REL_TOL, "verified": not reasons, "reasons": reasons}
    return {"flowsheet_id": flowsheet_id, "extracted_streams": tab["rows"], "unit_balance": tab["checks"],
            "current_utilities": {"hot_W": hot_now, "cold_W": cold_now, "by_unit": utilities,
                                  "recovered_in_exchangers_W": recovered},
            "streams_crossing_pinch": crossing,
            "linear_cp_targets": {"Q_hot_min_W": lin["Q_hot_min_W"], "Q_cold_min_W": lin["Q_cold_min_W"],
                                  "pinch": lin["pinch"]},
            "assumptions": [
                "Temperaturas de suministro y objetivo = las que tiene hoy el flowsheet (entrada del primer equipo "
                "térmico y salida del último de cada corriente); las intermedias entre equipos no son requisitos.",
                f"CP linealizado por tramos: H(T) de DWSIM (mismo paquete, misma composición y flujo) aproximado por "
                f"segmentos con error ≤ {linearization_tol_pct} % de la carga de cada corriente; los extremos son las "
                "entalpías del flowsheet.",
                "Presión interpolada linealmente con T entre entrada y salida de cada equipo para los flashes.",
                f"Cambio de fase: se localiza por bisección hasta {X.MIN_SPAN_K} K; un equipo isotérmico "
                f"(ΔT < {X.ISOTHERMAL_DT_K} K) se representa como un tramo de {X.ISOTHERMAL_SPAN_K} K con su calor latente.",
                "Servicio actual = calentadores (caliente) + enfriadores (frío); DWSIM no define el nivel del servicio "
                "(vapor, agua de enfriamiento): elegirlo con la gran curva compuesta.",
                f"ΔTmin = {dT_min_K} K dado por el usuario; contribución ΔTmin/2 por corriente salvo dT_contribution_K.",
                "Solo calentadores, enfriadores e intercambiadores; condensadores, rehervidores y camisas de reactor "
                "no se extraen."],
            **res}


ENERGY_TOOLS = [pinch_analysis, pinch_from_flowsheet]
