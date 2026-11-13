"""Agent / MCP tool of bloque M13: `pipe_network`, steady incompressible pipe networks with pumps and valves.

Pipe data, demands, elevations, boundary heads, pump curves, efficiencies, NPSHr, atmospheric and vapour pressure are
the USER's: missing ones raise, never default. The only design defaults are roughness by material name, read from the
`fluids` tables with their source and listed in `supuestos`.
"""
from __future__ import annotations

import copy
import difflib
import math

import numpy as np

from packages.network import solver as S
from packages.transport import hydraulics as HY
from packages.transport.flow_devices import fittings_K

METHOD = ("Gradiente global de Todini y Pilati (1988), el método de EPANET: Newton simultáneo sobre caudales y cargas; "
          "contraste independiente con Hardy Cross (1936). Fricción Darcy-Weisbach con el mismo Colebrook/Clamond "
          "de `fluids` que pipe_hydraulics (64/Re bajo Re 2300).")


class MissingData(ValueError):
    pass


def _need(where: str, d: dict, *keys: str) -> None:
    miss = [k for k in keys if d.get(k) is None]
    if miss:
        raise MissingData(f"{where}: faltan datos del usuario (no se suponen): {', '.join(miss)}")


def _fluid(fluid: dict, notes: list) -> tuple[S.Fluid, float | None]:
    if fluid.get("compound"):
        _need("fluid", fluid, "temperature_K")
        from thermo import Chemical
        c = Chemical(fluid["compound"], T=float(fluid["temperature_K"]))
        if c.rhol is None or c.mul is None:
            raise MissingData(f"thermo no da propiedades de líquido para {fluid['compound']} a {fluid['temperature_K']} K")
        notes.append(f"Propiedades de {c.name} a {fluid['temperature_K']} K de `thermo` (líquido a 1 atm): "
                     f"ρ = {c.rhol:.2f} kg/m³, μ = {c.mul:.4g} Pa·s, Pv = {c.Psat:.4g} Pa")
        return S.Fluid(float(c.rhol), float(c.mul)), float(c.Psat) if c.Psat else None
    _need("fluid", fluid, "density_kg_m3", "viscosity_Pa_s")
    pv = fluid.get("vapor_pressure_Pa")
    return S.Fluid(float(fluid["density_kg_m3"]), float(fluid["viscosity_Pa_s"])), (float(pv) if pv is not None else None)


def _roughness(lk: dict, notes: list) -> float:
    if lk.get("roughness_m") is not None:
        return float(lk["roughness_m"])
    mat = lk.get("material")
    if not mat:
        raise MissingData(f"tubo {lk['id']}: falta roughness_m o material (no se supone)")
    import fluids.friction as fr
    if mat.lower() in ("commercial steel", "acero comercial"):
        eps, src = HY.STEEL_ROUGHNESS_M, "acero comercial 4.57e-5 m (diagrama de Moody, el mismo de pipe_hydraulics)"
    elif mat in fr._roughness:
        eps, src = fr._roughness[mat], "tabla de tubos limpios de `fluids` (_roughness)"
    elif mat in fr.HHR_roughness:
        eps, src = float(fr.material_roughness(mat)), "Idel'chik y Ginevskiĭ 2007 vía `fluids` (promedio de su rango)"
    else:
        near = difflib.get_close_matches(mat, ["commercial steel", *fr._roughness, *fr.HHR_roughness], n=5, cutoff=0.3)
        raise MissingData(f"tubo {lk['id']}: material '{mat}' no está en las tablas de `fluids`; parecidos: {near} "
                          "(o dar roughness_m)")
    notes.append(f"Rugosidad del tubo {lk['id']} por material '{mat}': {eps:.3g} m — valor de diseño de {src}")
    return eps


def _diameter(lk: dict, notes: list) -> float:
    if lk.get("diameter_m") is not None:
        return float(lk["diameter_m"])
    if lk.get("nps_in") is not None:
        _need(f"tubo {lk['id']}", lk, "schedule")
        from fluids.piping import nearest_pipe
        NPS, Di, Do, t = nearest_pipe(NPS=float(lk["nps_in"]), schedule=str(lk["schedule"]))
        notes.append(f"Tubo {lk['id']}: NPS {NPS} cédula {lk['schedule']} → Di = {Di:.4f} m (ASME B36.10M vía fluids)")
        return float(Di)
    raise MissingData(f"tubo/válvula {lk['id']}: falta diameter_m (o nps_in + schedule)")


def _interp(points, q: float):
    pts = sorted((float(a), float(b)) for a, b in points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(q, xs, ys)), not (xs[0] - 1e-12 <= q <= xs[-1] + 1e-12)


def build_network(nodes: list[dict], links: list[dict], fluid: dict) -> tuple[S.Network, dict]:
    notes: list[str] = []
    fl, pv = _fluid(fluid, notes)
    J, F, meta = {}, {}, {"nodes": {}, "links": {}, "vapor_pressure_Pa": pv, "notes": notes}
    for n in nodes:
        nid, t = str(n.get("id")), n.get("type")
        meta["nodes"][nid] = n
        if t == "junction":
            if n.get("demand_m3_s") is None and n.get("demand_kg_s") is not None:
                n = {**n, "demand_m3_s": float(n["demand_kg_s"]) / fl.rho}
            _need(f"nodo {nid}", n, "elevation_m", "demand_m3_s")
            J[nid] = {"z": float(n["elevation_m"]), "d": float(n["demand_m3_s"])}
        elif t == "reservoir":
            _need(f"nodo {nid}", n, "head_m")
            F[nid] = {"z": float(n.get("elevation_m", n["head_m"])), "H": float(n["head_m"])}
        elif t == "tank":
            _need(f"nodo {nid}", n, "elevation_m", "level_m")
            F[nid] = {"z": float(n["elevation_m"]), "H": float(n["elevation_m"]) + float(n["level_m"])}
        elif t == "pressure":
            _need(f"nodo {nid}", n, "elevation_m", "pressure_Pa_g")
            F[nid] = {"z": float(n["elevation_m"]), "H": float(n["elevation_m"]) + float(n["pressure_Pa_g"]) / (fl.rho * S.G)}
        else:
            raise MissingData(f"nodo {nid}: type debe ser junction | reservoir | tank | pressure")
    if len(J) + len(F) != len(nodes):
        raise MissingData("ids de nodo repetidos")
    out = []
    for lk in links:
        lid, t = str(lk.get("id")), lk.get("type")
        a, b = str(lk.get("from")), str(lk.get("to"))
        for x in (a, b):
            if x not in J and x not in F:
                raise MissingData(f"enlace {lid}: nodo '{x}' no existe")
        meta["links"][lid] = lk
        if t == "pipe" and lk.get("resistance"):
            rs = lk["resistance"]
            _need(f"tubo {lid} (resistencia)", rs, "K", "n")
            notes.append(f"Tubo {lid}: resistencia genérica h = K·Q^n del usuario (K = {rs['K']}, n = {rs['n']}, SI); "
                         "no usa Colebrook")
            D = float(lk["diameter_m"]) if lk.get("diameter_m") is not None else 0.0
            out.append(S.Link(lid, "pipe", a, b, D=D, res_K=float(rs["K"]), res_n=float(rs["n"]),
                              cv=bool(lk.get("check_valve"))))
        elif t == "pipe":
            _need(f"tubo {lid}", lk, "length_m")
            D = _diameter(lk, notes)
            hw = lk.get("hazen_williams_C")
            eps = 0.0 if hw is not None else _roughness(lk, notes)
            if hw is not None:
                notes.append(f"Tubo {lid}: Hazen-Williams C = {hw} (constante SI de EPANET 2.2); solo agua fría, "
                             "no es el Colebrook de pipe_hydraulics")
            Kf = None
            if lk.get("fittings"):
                fits = lk["fittings"]
                fittings_K(fits, D, 1e5)                      # validates the names now
                Kf = (lambda Re, fits=fits, D=D: fittings_K(fits, D, Re)["K_total"])
            out.append(S.Link(lid, "pipe", a, b, D=D, L=float(lk["length_m"]), eps=eps,
                              K=float(lk.get("minor_loss_K") or 0.0), K_of_Re=Kf, cv=bool(lk.get("check_valve")),
                              hw_C=float(hw) if hw is not None else None))
        elif t == "pump":
            if not lk.get("curve_points") and not lk.get("curve"):
                raise MissingData(f"bomba {lid}: falta la curva (curve_points [[q_m3_s, h_m], …] o curve {{h0_m, r, n}})")
            pc = S.fit_pump_curve(lk.get("curve_points"), lk.get("curve"), float(lk.get("speed_ratio") or 1.0))
            out.append(S.Link(lid, "pump", a, b, pump=pc))
        elif t in ("prv", "fcv"):
            D = _diameter(lk, notes)
            if t == "prv":
                if lk.get("setting_head_m") is not None:
                    s = float(lk["setting_head_m"])
                elif lk.get("setting_pressure_Pa_g") is not None:
                    if b not in J:
                        raise MissingData(f"PRV {lid}: el nodo aguas abajo debe ser una unión")
                    s = J[b]["z"] + float(lk["setting_pressure_Pa_g"]) / (fl.rho * S.G)
                else:
                    raise MissingData(f"PRV {lid}: falta setting_pressure_Pa_g (o setting_head_m)")
            else:
                _need(f"FCV {lid}", lk, "setting_flow_m3_s")
                s = float(lk["setting_flow_m3_s"])
            out.append(S.Link(lid, t, a, b, D=D, K=float(lk.get("minor_loss_K") or 0.0), setting=s))
        else:
            raise MissingData(f"enlace {lid}: type debe ser pipe | pump | prv | fcv")
    if len({lk.id for lk in out}) != len(out):
        raise MissingData("ids de enlace repetidos")
    return S.Network(fl, J, F, out), meta


def _regime(Re: float) -> str:
    return "laminar" if Re < 2300.0 else ("transición" if Re < 4000.0 else "turbulento")


def _system_curve(net: S.Network, group: list[S.Link], q_duty: float, n_pts: int) -> list[dict]:
    """Head the rest of the network asks of a pump station (pumps between the same two nodes) versus its total flow:
    the station is replaced by a fixed-flow element and the network is re-solved at each flow."""
    a, b = group[0].a, group[0].b
    qmax = max((lk.pump.q_range[1] * lk.pump.w for lk in group if math.isfinite(lk.pump.q_range[1])), default=0.0)
    top = max(1.5 * q_duty, qmax * len(group) if q_duty <= 0 else 0.0, 1e-6)
    flows = sorted({*np.linspace(0.0, top, n_pts).tolist(), q_duty})
    ids = {lk.id for lk in group}
    pts = []
    for Q in flows:
        n2 = copy.deepcopy(net)
        n2.links = [lk for lk in n2.links if lk.id not in ids]
        n2.links.append(S.Link("__estacion__", "fixed", a, b, setting=Q, status="active"))
        try:
            r = S.solve(n2, q0={"__estacion__": Q})
            Ha = r["H"].get(a, net.fixed.get(a, {}).get("H"))
            Hb = r["H"].get(b, net.fixed.get(b, {}).get("H"))
            pts.append({"caudal_m3_s": Q, "carga_requerida_m": (Hb - Ha) if None not in (Ha, Hb) else None})
        except S.NetworkError as e:
            pts.append({"caudal_m3_s": Q, "carga_requerida_m": None, "error": str(e)})
    return pts


def _combined_curve(group: list[S.Link], n_pts: int = 8) -> list[dict]:
    """Parallel pumps: at equal head the flows add (each pump on its own curve, zero flow above its shut-off)."""
    from scipy.optimize import brentq
    h_top = max(lk.pump.shutoff for lk in group)
    rows = []
    for h in np.linspace(h_top, 0.0, n_pts):
        tot = 0.0
        for lk in group:
            pc = lk.pump
            if h >= pc.shutoff:
                continue
            hi = pc.q_range[1] * pc.w if math.isfinite(pc.q_range[1]) else 10.0
            while pc.gain(hi)[0] > h and hi < 1e3:
                hi *= 2.0
            tot += brentq(lambda q: pc.gain(q)[0] - h, 0.0, hi)
        rows.append({"carga_m": float(h), "caudal_total_m3_s": tot})
    return rows


def pipe_network(nodes: list[dict], links: list[dict], fluid: dict, atmospheric_pressure_Pa: float | None = None,
                 system_curve: bool = True, system_curve_points: int = 11, hardy_cross_check: bool = True) -> dict:
    try:
        net, meta = build_network(nodes, links, fluid)
        sol = S.solve(net)
    except (MissingData, S.NetworkError) as e:
        return {"ok": False, "error": str(e)}
    fl, q, H = net.fluid, sol["q"], sol["H"]
    known = {n: v["H"] for n, v in net.fixed.items()}
    head = lambda m: H[m] if H.get(m) is not None else known.get(m)
    notes, warns = list(meta["notes"]), []
    rg = fl.rho * S.G
    pipes, pumps, valves = [], [], []
    same_as_single, groups = 0.0, {}
    for lk in net.links:
        Q = q[lk.id]
        if lk.kind == "pipe":
            h, _, x = S.pipe_loss(lk, Q, fl)
            row = {"id": lk.id, "de": lk.a, "a": lk.b, "caudal_m3_s": Q, "caudal_kg_s": Q * fl.rho,
                   "velocidad_m_s": Q / lk.area if lk.D > 0 else None, "Re": x["Re"],
                   "regimen": _regime(x["Re"]) if x["Re"] is not None else None,
                   "f_darcy": x["f"], "K_accesorios": x["K"], "perdida_carga_m": h, "dP_friccion_Pa": h * rg,
                   "estado": sol["status"][lk.id],
                   "friccion": ("resistencia K·Q^n" if lk.res_K is not None else
                                "Hazen-Williams" if lk.hw_C else "Darcy-Weisbach Colebrook/Clamond")}
            pipes.append(row)
            if lk.hw_C is None and lk.res_K is None and abs(Q) > 0 and lk.status == "open":
                ref = HY.pipe_dp(fl.rho, fl.mu, abs(Q) * fl.rho, lk.D, lk.L, lk.eps, 0.0, x["K"])
                same_as_single = max(same_as_single, abs(ref["dP_friction_Pa"] + ref["dP_fittings_Pa"] - abs(h) * rg)
                                     / max(abs(h) * rg, 1e-30))
            if x["Re"] is not None and 2300.0 <= x["Re"] < 4000.0:
                warns.append(f"tubo {lk.id}: Re = {x['Re']:.0f} en transición; f salta de 64/Re a Colebrook en 2300 "
                             "(igual que pipe_hydraulics)")
        elif lk.kind == "pump":
            groups.setdefault((lk.a, lk.b), []).append(lk)
        else:
            h = head(lk.a) - head(lk.b) if None not in (head(lk.a), head(lk.b)) else None
            valves.append({"id": lk.id, "tipo": lk.kind.upper(), "de": lk.a, "a": lk.b, "caudal_m3_s": Q,
                           "estado": sol["status"][lk.id], "perdida_carga_m": h,
                           "ajuste": ({"carga_m": lk.setting,
                                       "presion_Pa_g": (lk.setting - net.junctions[lk.b]["z"]) * rg
                                       if lk.b in net.junctions else None} if lk.kind == "prv"
                                      else {"caudal_m3_s": lk.setting})})
    pv, Patm = meta["vapor_pressure_Pa"], atmospheric_pressure_Pa
    for lk in [x for x in net.links if x.kind == "pump"]:
        um = meta["links"][lk.id]
        Q = q[lk.id]
        dh = head(lk.b) - head(lk.a)
        row = {"id": lk.id, "de": lk.a, "a": lk.b, "estado": sol["status"][lk.id], "caudal_m3_s": Q,
               "carga_m": dh if lk.status == "open" else 0.0, "curva": lk.pump.source, "relacion_velocidad": lk.pump.w,
               "potencia_hidraulica_kW": rg * Q * dh / 1e3 if lk.status == "open" else 0.0}
        lo, hi = lk.pump.q_range
        if lk.status == "open" and (Q > hi * lk.pump.w + 1e-12 or Q < lo * lk.pump.w - 1e-12):
            warns.append(f"bomba {lk.id}: opera a {Q:.4g} m³/s, fuera del rango de su curva dada "
                         f"({lo * lk.pump.w:.4g}–{hi * lk.pump.w:.4g} m³/s): extrapolación")
        eta = None
        if um.get("efficiency_points"):
            eta, out = _interp(um["efficiency_points"], Q / lk.pump.w)
            if out:
                warns.append(f"bomba {lk.id}: eficiencia extrapolada (caudal fuera de sus puntos)")
        elif um.get("efficiency") is not None:
            eta = float(um["efficiency"])
        row["eficiencia"] = eta
        row["potencia_eje_kW"] = row["potencia_hidraulica_kW"] / eta if eta else None
        if eta is None:
            row["nota_potencia"] = "sin eficiencia del usuario: solo potencia hidráulica"
        # NPSH available at the suction node (pump centre line = suction node elevation unless given)
        npsh = {"NPSHa_m": None}
        z_c = um.get("centerline_elevation_m")
        nd = meta["nodes"][lk.a]
        z_c = float(z_c) if z_c is not None else (net.junctions[lk.a]["z"] if lk.a in net.junctions
                                                 else (net.fixed[lk.a]["z"] if nd.get("type") != "reservoir" or
                                                       nd.get("elevation_m") is not None else None))
        missing = [k for k, v in (("atmospheric_pressure_Pa", Patm), ("vapor_pressure_Pa", pv),
                                  ("centerline_elevation_m", z_c)) if v is None]
        if lk.status == "open" and not missing:
            # Node heads of a network model are energy heads (velocity heads neglected at nodes, as in EPANET), so the
            # suction energy head is H_a itself: NPSHa = (H_a − z_eje) + (Patm − Pv)/(ρg) — the textbook
            # Patm/ρg + Δz − h_L − Pv/ρg with the losses already inside H_a.
            npsh = {"NPSHa_m": (head(lk.a) - z_c) + (Patm - pv) / rg, "carga_succion_m": head(lk.a),
                    "carga_succion_manometrica_m": head(lk.a) - z_c,
                    "elevacion_eje_m": z_c, "Patm_Pa": Patm, "Pv_Pa": pv}
            nr = None
            if um.get("npshr_points"):
                nr, out = _interp(um["npshr_points"], Q / lk.pump.w)
                nr *= lk.pump.w ** 2
                if out:
                    warns.append(f"bomba {lk.id}: NPSHr extrapolado")
            elif um.get("npshr_m") is not None:
                nr = float(um["npshr_m"])
            npsh["NPSHr_m"] = nr
            if nr is not None:
                npsh["margen_m"] = npsh["NPSHa_m"] - nr
                npsh["cavita"] = npsh["NPSHa_m"] < nr
                if npsh["cavita"]:
                    warns.append(f"bomba {lk.id}: NPSHa {npsh['NPSHa_m']:.2f} m < NPSHr {nr:.2f} m → cavitación")
            else:
                npsh["nota"] = "sin NPSHr del usuario: no se juzga el margen"
        elif lk.status == "open":
            npsh["nota"] = "NPSHa sin calcular; faltan datos del usuario: " + ", ".join(missing)
        row["npsh"] = npsh
        pumps.append(row)
    stations = []
    for (a, b), grp in groups.items():
        open_ = [lk for lk in grp if lk.status == "open"]
        Qd = sum(q[lk.id] for lk in grp)
        st = {"bombas": [lk.id for lk in grp], "de": a, "a": b,
              "arreglo": "paralelo" if len(grp) > 1 else "única", "caudal_total_m3_s": Qd,
              "carga_m": head(b) - head(a), "bombas_operando": len(open_)}
        if len(grp) > 1:
            st["curva_combinada"] = _combined_curve(grp)
        if system_curve and open_:
            sc = _system_curve(net, grp, Qd, system_curve_points)
            st["curva_sistema"] = sc
            at = [p for p in sc if abs(p["caudal_m3_s"] - Qd) < 1e-15 and p["carga_requerida_m"] is not None]
            if at:
                st["cruce_curva_sistema_m"] = abs(at[0]["carga_requerida_m"] - st["carga_m"])
        stations.append(st)
    series = [(s1["bombas"], s2["bombas"]) for s1 in stations for s2 in stations if s1["a"] == s2["de"]
              and s1["a"] in net.junctions and net.junctions[s1["a"]]["d"] == 0 and
              sum(1 for lk in net.links if s1["a"] in (lk.a, lk.b)) == 2]
    res = S.node_residuals(net, q)
    nodes_out = []
    for nid, n in meta["nodes"].items():
        if nid in net.junctions:
            z, Hn = net.junctions[nid]["z"], H.get(nid)
            nodes_out.append({"id": nid, "tipo": "unión", "elevacion_m": z, "carga_m": Hn,
                              "presion_Pa_g": (Hn - z) * rg if Hn is not None else None,
                              "presion_bar_g": (Hn - z) * rg / 1e5 if Hn is not None else None,
                              "demanda_m3_s": net.junctions[nid]["d"], "residuo_masa_m3_s": res[nid],
                              "residuo_masa_kg_s": res[nid] * fl.rho})
            if Hn is None:
                warns.append(f"nodo {nid}: aislado por enlaces cerrados, sin carga definida")
            elif Hn - z < 0:
                warns.append(f"nodo {nid}: presión manométrica negativa ({(Hn - z) * rg:.0f} Pa)")
        else:
            out_flow = sum(q[lk.id] for lk in net.links if lk.a == nid) - sum(q[lk.id] for lk in net.links if lk.b == nid)
            z = net.fixed[nid]["z"]
            nodes_out.append({"id": nid, "tipo": {"reservoir": "reservorio", "tank": "tanque", "pressure": "presión fija"}[n["type"]],
                              "elevacion_m": z, "carga_m": net.fixed[nid]["H"],
                              "presion_Pa_g": (net.fixed[nid]["H"] - z) * rg,
                              "suministro_m3_s": out_flow, "nota": "suministro > 0 entrega a la red; < 0 recibe"})
    max_res = max((abs(v) for v in res.values()), default=0.0)
    en = S.energy_residuals(net, q, H)
    max_en = max(en.values(), default=0.0)
    hc = {"aplica": False, "motivo": "no solicitado"}
    if hardy_cross_check:
        n3 = copy.deepcopy(net)
        n3.links = [lk for lk in n3.links if lk.status != "closed"]
        r3 = S.hardy_cross(n3)
        if r3["aplica"]:
            hc = {"aplica": True, "lazos": r3["loops"], "iteraciones": r3["iterations"], "convergio": r3["converged"],
                  "max_dif_caudal_m3_s": max((abs(r3["q"][k] - q[k]) for k in r3["q"]), default=0.0),
                  "max_dif_carga_m": max((abs(r3["H"][k] - H[k]) for k in net.junctions if H.get(k) is not None),
                                         default=0.0)}
        else:
            hc = r3
    verified = (max_res <= S.MASS_TOL_M3_S and max_en <= S.ENERGY_TOL_M and same_as_single < 1e-9 and
                (not hc.get("aplica") or (hc["max_dif_caudal_m3_s"] < 1e-6 and hc["max_dif_carga_m"] < 1e-4)))
    assumptions = [
        "Flujo estacionario, incompresible e isotérmico (una sola ρ y μ para toda la red).",
        "Tanques y reservorios a carga fija (foto instantánea; sin simulación en el tiempo).",
        "La elevación del nodo de succión es la del eje de la bomba, salvo centerline_elevation_m.",
        "Las cargas de nodo son cargas de energía con la carga de velocidad despreciada en los nodos (convención de "
        "EPANET): presión = ρg(H − z) y NPSHa = (H_succión − z_eje) + (Patm − Pv)/(ρg).",
        "Válvulas abiertas (PRV/FCV en estado abierto) con la K que da el usuario; sin K, pérdida casi nula.",
        "Tubos sin accesorios declarados no tienen pérdidas menores.",
        *notes]
    return {"ok": True, "metodo": METHOD, "tuberias": pipes, "bombas": pumps, "estaciones_bombeo": stations,
            "bombas_en_serie": series, "valvulas": valves, "nodos": nodes_out,
            "evidencia": {"residuo_masa_max_m3_s": max_res, "tolerancia_masa_m3_s": S.MASS_TOL_M3_S,
                          "residuo_energia_max_m": max_en, "tolerancia_energia_m": S.ENERGY_TOL_M,
                          "iteraciones_newton": sol["newton_iterations"], "pasadas_estado": sol["status_passes"],
                          "cambios_estado": sol["status_log"], "hardy_cross": hc,
                          "friccion_vs_pipe_hydraulics_dif_rel_max": same_as_single, "verified": verified},
            "supuestos": assumptions, "avisos": warns}
