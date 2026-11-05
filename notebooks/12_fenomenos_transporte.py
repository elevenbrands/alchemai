"""Notebook 12 — transport phenomena tools vs DWSIM 10 (bloque 18).

A) Pipe ΔP: `pipe_hydraulics` (Darcy-Weisbach + Clamond/Colebrook on DWSIM's own stream properties) vs the DWSIM
   Pipe Segment unit (Beggs-Brill, adiabatic, commercial steel) on the same stream.
B) Heat-exchanger area: DWSIM HeatExchanger in CalcBothTemp_UA with U and A given; `heat_exchanger_area` must
   recover A from DWSIM's Q and outlet temperatures.
(A cross-check of the film coefficient against DWSIM's pipe h_internal was tried and dropped: with the estimated
thermal profile DWSIM reported h_internal = 0 and cooled the water to ambient in 50 m.)
Usage: PYTHONPATH=. uv run python notebooks/12_fenomenos_transporte.py [A|B]   (prints one JSON line per case)
"""
import json
import sys

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_transport import heat_exchanger_area, pipe_hydraulics, tube_film_coefficient  # noqa: E402

SCH40 = {2: 0.05250, 3: 0.07793, 4: 0.10226, 6: 0.15405}
PIPES = [  # id, compounds, package, T_K, P_Pa, kg/s, fractions, NPS, L, dz
    ("P1 agua 4 in", ["Water"], "SteamTables", 293.15, 5e5, 10.0, {"Water": 1.0}, 4, 100.0, 0.0),
    ("P2 agua 60 °C 2 in, sube 10 m", ["Water"], "SteamTables", 333.15, 6e5, 3.0, {"Water": 1.0}, 2, 200.0, 10.0),
    ("P3 agua 6 in 500 m", ["Water"], "SteamTables", 293.15, 10e5, 40.0, {"Water": 1.0}, 6, 500.0, 0.0),
    ("P4 n-hexano líquido 3 in", ["N-hexane"], "PengRobinson", 300.0, 5e5, 15.0, {"N-hexane": 1.0}, 3, 150.0, 0.0),
    ("P5 metano gas 20 bar 6 in", ["Methane"], "PengRobinson", 300.0, 20e5, 5.0, {"Methane": 1.0}, 6, 200.0, 0.0),
    ("P6 glicerol laminar 2 in", ["Glycerol"], "PengRobinson", 300.0, 3e5, 0.5, {"Glycerol": 1.0}, 2, 50.0, 0.0),
]


def _pkg(name):
    return name


def case_pipe(c):
    cid, comps, pkg, Tk, P, m, x, nps, L, dz = c
    fid = T.new_flowsheet(cid, comps, _pkg(pkg))["flowsheet_id"]
    T.add_material_stream(fid, "in", temperature_K=Tk, pressure_Pa=P, mass_flow_kg_s=m, mole_fractions=x)
    T.add_material_stream(fid, "out")
    T.add_unit(fid, "pipe", "L-1", {"length_m": L, "inner_diameter_m": SCH40[nps], "elevation_change_m": dz, "increments": 20})
    T.connect(fid, "L-1", feeds=["in"], products=["out"])
    r = T.solve(fid)
    u = T.read_unit(fid, "L-1")
    ours = pipe_hydraulics(fid, "in", L, SCH40[nps], elevation_change_m=dz)
    out = {"case": cid, "solved": r["solved"], "Re": ours["Re"], "regime": ours["regime"], "v_m_s": ours["velocity_m_s"],
           "dwsim_v_m_s": u.get("liquid_velocity_m_s"), "dP_fric_ours": ours["dP_friction_Pa"], "dP_fric_dwsim": u["dP_friction_Pa"],
           "dP_static_ours": ours["dP_static_Pa"], "dP_static_dwsim": u["dP_static_Pa"], "regime_dwsim": u.get("flow_regime"),
           "rho": ours["properties"]["density_kg_m3"], "mu": ours["properties"]["viscosity_Pa_s"]}
    out["rel_err_fric"] = (out["dP_fric_ours"] - out["dP_fric_dwsim"]) / out["dP_fric_dwsim"]
    return out


HX = [  # id, compounds, pkg, hot (T, P, kg/s, x), cold (T, P, kg/s, x), U, A
    ("H1 agua/agua", ["Water"], "SteamTables", (360.0, 3e5, 8.0, {"Water": 1.0}), (300.0, 4e5, 20.0, {"Water": 1.0}), 850.0, 40.0),
    ("H2 n-hexano/agua", ["N-hexane", "Water"], "PengRobinson", (380.0, 6e5, 10.0, {"N-hexane": 1.0}),
     (298.0, 4e5, 12.0, {"Water": 1.0}), 450.0, 60.0),
]


def case_hx(c):
    from DWSIM.UnitOperations.UnitOperations import HeatExchangerCalcMode
    cid, comps, pkg, hot, cold, U, A = c
    fid = T.new_flowsheet(cid, comps, _pkg(pkg))["flowsheet_id"]
    for n, (Tk, P, m, x) in (("hi", hot), ("ci", cold)):
        T.add_material_stream(fid, n, temperature_K=Tk, pressure_Pa=P, mass_flow_kg_s=m, mole_fractions={k: x.get(k, 0.0) for k in comps})
    T.add_material_stream(fid, "ho"); T.add_material_stream(fid, "co")
    s = T._s(fid)
    b = (s["fs"].AddHeatExchanger("E-1").WithCalculationMode(HeatExchangerCalcMode.CalcBothTemp_UA).WithGlobalUA(U * A).WithExchangeArea(A))   # Fluent API: GlobalUA is U·A in W/K
    s["units"]["E-1"], s["kinds"]["E-1"], s["params"]["E-1"] = b, "heat_exchanger", {"UA_W_K": U * A}
    T.connect(fid, "E-1", feeds=["hi", "ci"], products=["ho", "co"])
    r = T.solve(fid)
    o = getattr(b.Object, "__implementation__", b.Object)
    Q = float(o.Q)
    sv = r["streams"]
    ours = heat_exchanger_area(Q, sv["hi"]["T_K"], sv["ho"]["T_K"], sv["ci"]["T_K"], sv["co"]["T_K"], U)
    flow = str(getattr(o, "FlowDir", "?"))
    return {"case": cid, "solved": r["solved"], "flow_dir": flow, "Q_kW": Q, "T_hot_out": sv["ho"]["T_K"], "T_cold_out": sv["co"]["T_K"],
            "LMTD_ours": ours["LMTD_K"], "A_given": A, "A_ours": ours["area_m2"], "rel_err_A": (ours["area_m2"] - A) / A,
            "energy_rel_err": max(u["energy_rel_err"] for u in r["units"])}


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "A"
    jobs = {"A": [lambda c=c: case_pipe(c) for c in PIPES], "B": [lambda c=c: case_hx(c) for c in HX]}[part]
    for j in jobs:
        try:
            print(json.dumps(j(), ensure_ascii=False), flush=True)
        except Exception as ex:  # report and keep going
            print(json.dumps({"error": f"{type(ex).__name__}: {ex}"[:400]}), flush=True)
