"""Reference runs of EPANET 2.2 (public domain / MIT toolkit) through WNTR (Sandia, Revised BSD), run as a SEPARATE
process with its own throw-away environment, never imported by AlchemAI:

    uv run --no-project --python 3.12 --with wntr==1.5.0 --with six python packages/network/epanet_reference.py

Writes tests/data/epanet_reference.json: for each variant of the EPANET example network Net1 (snapshot at t = 0,
tank 2 at its initial level) the network in SI units as EPANET read it and EPANET's results (heads, pressures, flows,
link status). The test suite rebuilds each network for `pipe_network` from that JSON and compares.
"""
from __future__ import annotations

import json
import os
import sys

import wntr

NU_EPANET = 1.1e-5 * 0.09290304          # EPANET's kinematic viscosity at relative viscosity 1.0 (ft²/s → m²/s)
GPM = 6.30901964e-5                       # m³/s per US gpm
FT = 0.3048
NET1 = os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net1.inp")


def _dump(wn, res, name: str, note: str) -> dict:
    nodes, links = {}, {}
    for n, j in wn.junctions():
        nodes[n] = {"type": "junction", "elevation_m": j.elevation, "demand_m3_s": j.demand_timeseries_list[0].base_value}
    for n, r in wn.reservoirs():
        nodes[n] = {"type": "reservoir", "head_m": r.base_head, "elevation_m": r.base_head}
    for n, t in wn.tanks():
        nodes[n] = {"type": "tank", "elevation_m": t.elevation, "init_level_m": t.init_level,
                    "head_m": t.elevation + t.init_level}
    for n, p in wn.pipes():
        links[n] = {"type": "pipe", "from": p.start_node_name, "to": p.end_node_name, "length_m": p.length,
                    "diameter_m": p.diameter, "roughness": p.roughness, "minor_loss": p.minor_loss,
                    "check_valve": bool(p.check_valve)}
    for n, p in wn.pumps():
        c = wn.get_curve(p.pump_curve_name)
        links[n] = {"type": "pump", "from": p.start_node_name, "to": p.end_node_name, "curve_points": c.points,
                    "speed": p.base_speed}
    for n, v in wn.valves():
        links[n] = {"type": "valve", "valve_type": v.valve_type, "from": v.start_node_name, "to": v.end_node_name,
                    "diameter_m": v.diameter, "setting": v.initial_setting, "minor_loss": v.minor_loss}
    head = res.node["head"].iloc[0]
    pres = res.node["pressure"].iloc[0]
    flow = res.link["flowrate"].iloc[0]
    status = res.link["status"].iloc[0]
    return {"name": name, "note": note, "headloss": wn.options.hydraulic.headloss,
            "kinematic_viscosity_m2_s": NU_EPANET * wn.options.hydraulic.viscosity,
            "specific_gravity": wn.options.hydraulic.specific_gravity, "nodes": nodes, "links": links,
            "epanet": {"head_m": {k: float(v) for k, v in head.items()},
                       "pressure_m": {k: float(v) for k, v in pres.items()},
                       "flow_m3_s": {k: float(v) for k, v in flow.items()},
                       "status": {k: int(v) for k, v in status.items()}}}


def _run(wn, name: str, note: str, tmp: str) -> dict:
    wn.options.time.duration = 0
    wn.options.hydraulic.accuracy = 1e-6          # tighter than Net1's 0.001 so the comparison is not limited by it
    wn.options.hydraulic.trials = 200
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, name))
    return _dump(wn, res, name, note)


def _dw(wn, eps_m: float):
    wn.options.hydraulic.headloss = "D-W"
    for _, p in wn.pipes():
        p.roughness = eps_m                        # WNTR keeps D-W roughness in m (writes millifeet to the .inp)
    return wn


def main(out: str, tmp: str) -> None:
    variants = []
    wn = wntr.network.WaterNetworkModel(NET1)
    variants.append(_run(wn, "net1_hw", "Net1 tal cual (Hazen-Williams C=100), t = 0", tmp))

    wn = _dw(wntr.network.WaterNetworkModel(NET1), 1.5e-4)
    variants.append(_run(wn, "net1_dw", "Net1 con Darcy-Weisbach, ε = 0.15 mm en todos los tubos", tmp))

    wn = _dw(wntr.network.WaterNetworkModel(NET1), 1.5e-4)
    wn.add_pump("9b", "9", "10", "HEAD", "1")
    variants.append(_run(wn, "net1_dw_parallel", "net1_dw + una segunda bomba igual en paralelo (9b)", tmp))

    wn = _dw(wntr.network.WaterNetworkModel(NET1), 1.5e-4)
    wn.get_link("110").check_valve = True
    wn.get_link("110").initial_status = wntr.network.LinkStatus.Opened
    variants.append(_run(wn, "net1_dw_cv110", "net1_dw con válvula check en el tubo 110 (solo 2 → 12)", tmp))

    for vt, setting, tag in (("PRV", None, "prv"), ("FCV", 0.004, "fcv")):
        wn = _dw(wntr.network.WaterNetworkModel(NET1), 1.5e-4)
        p = wn.get_link("111")
        a, b, d = p.start_node_name, p.end_node_name, p.diameter
        wn.remove_link("111")
        s = setting if setting is not None else 81.0           # PRV: 81 m of pressure at node 21 (between its closed and open heads, so it is ACTIVE)
        wn.add_valve("111v", a, b, d, vt, 0.0, s)
        variants.append(_run(wn, f"net1_dw_{tag}", f"net1_dw con el tubo 111 cambiado por una {vt} "
                             f"(ajuste {s} {'m de presión' if vt == 'PRV' else 'm³/s'})", tmp))
    wntr.network.write_inpfile(wn, os.path.join(tmp, "check.inp"))
    json.dump({"source": "EPANET 2.2 via WNTR " + wntr.__version__, "net1_inp": "wntr/library/networks/Net1.inp",
               "variants": variants}, open(out, "w"), indent=1)
    print("wrote", out, [v["name"] for v in variants])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data/epanet_reference.json",
         sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TMPDIR", "/tmp"))
