"""Bloque M13: pipe networks with pumps and valves (`pipe_network`).

Evidence: EPANET 2.2 (run as a separate process through WNTR, see packages/network/epanet_reference.py) on its example
network Net1 and five variants; the single-pipe limit against pipe_hydraulics; closed-form pumps in parallel and in
series; NPSHa by hand; Hardy Cross versus the global gradient; continuity at every node.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from packages.network import solver as S
from packages.network.tool import pipe_network
from packages.transport import hydraulics as HY

REF = json.load(open(os.path.join(os.path.dirname(__file__), "data", "epanet_reference.json")))
VAR = {v["name"]: v for v in REF["variants"]}
WATER = {"density_kg_m3": 998.2, "viscosity_Pa_s": 1.002e-3, "vapor_pressure_Pa": 2339.0}


def _from_epanet(v: dict) -> tuple[list, list, dict]:
    nodes, links = [], []
    for k, n in v["nodes"].items():
        if n["type"] == "junction":
            nodes.append({"id": k, "type": "junction", "elevation_m": n["elevation_m"], "demand_m3_s": n["demand_m3_s"]})
        elif n["type"] == "reservoir":
            nodes.append({"id": k, "type": "reservoir", "head_m": n["head_m"]})
        else:
            nodes.append({"id": k, "type": "tank", "elevation_m": n["elevation_m"], "level_m": n["init_level_m"]})
    for k, l in v["links"].items():
        if l["type"] == "pipe":
            d = {"id": k, "type": "pipe", "from": l["from"], "to": l["to"], "length_m": l["length_m"],
                 "diameter_m": l["diameter_m"], "minor_loss_K": l["minor_loss"], "check_valve": l["check_valve"]}
            d.update({"hazen_williams_C": l["roughness"]} if v["headloss"] == "H-W" else {"roughness_m": l["roughness"]})
            links.append(d)
        elif l["type"] == "pump":
            links.append({"id": k, "type": "pump", "from": l["from"], "to": l["to"], "curve_points": l["curve_points"]})
        else:
            d = {"id": k, "type": l["valve_type"].lower(), "from": l["from"], "to": l["to"],
                 "diameter_m": l["diameter_m"], "minor_loss_K": l["minor_loss"]}
            d.update({"setting_pressure_Pa_g": l["setting"] * 1000.0 * S.G} if l["valve_type"] == "PRV"
                     else {"setting_flow_m3_s": l["setting"]})
            links.append(d)
    fluid = {"density_kg_m3": 1000.0, "viscosity_Pa_s": v["kinematic_viscosity_m2_s"] * 1000.0}
    return nodes, links, fluid


def _run(v: dict, **kw) -> dict:
    nodes, links, fluid = _from_epanet(v)
    r = pipe_network(nodes, links, fluid, **kw)
    assert r["ok"], r
    return r


def _flows(r: dict) -> dict:
    out = {p["id"]: p["caudal_m3_s"] for p in r["tuberias"]}
    out.update({p["id"]: p["caudal_m3_s"] for p in r["bombas"]})
    out.update({p["id"]: p["caudal_m3_s"] for p in r["valvulas"]})
    return out


# ---------------------------------------------------------------------------------------------- EPANET Net1
def test_net1_hazen_williams_matches_epanet_exactly():
    """Same head-loss law as EPANET (H-W), so the only differences are the two solvers' tolerances."""
    v = VAR["net1_hw"]
    r = _run(v)
    e = v["epanet"]
    for k, Q in _flows(r).items():
        assert Q == pytest.approx(e["flow_m3_s"][k], abs=1e-6), k
    for n in r["nodos"]:
        assert n["carga_m"] == pytest.approx(e["head_m"][n["id"]], abs=1e-3), n["id"]
        if n["tipo"] == "unión":
            assert n["presion_Pa_g"] / (1000.0 * S.G) == pytest.approx(e["pressure_m"][n["id"]], abs=1e-3)
    assert r["evidencia"]["verified"]
    pump = r["bombas"][0]
    assert pump["caudal_m3_s"] == pytest.approx(0.1177, abs=1e-4)          # EPANET: 117.74 L/s at t = 0


@pytest.mark.parametrize("name", ["net1_dw", "net1_dw_parallel", "net1_dw_cv110", "net1_dw_prv", "net1_dw_fcv"])
def test_net1_darcy_weisbach_variants_vs_epanet(name):
    """EPANET's D-W uses Swamee-Jain (explicit) where AlchemAI uses Colebrook/Clamond: ≤ 0.15 % of the largest flow
    and ≤ 0.07 m of head apart; link statuses (check closed, PRV/FCV active) identical."""
    v = VAR[name]
    r = _run(v)
    e = v["epanet"]
    qmax = max(abs(x) for x in e["flow_m3_s"].values())
    for k, Q in _flows(r).items():
        assert abs(Q - e["flow_m3_s"][k]) <= 1.5e-3 * qmax, (k, Q, e["flow_m3_s"][k])
    for n in r["nodos"]:
        assert n["carga_m"] == pytest.approx(e["head_m"][n["id"]], abs=0.07), n["id"]
    st = {x["id"]: x["estado"] for x in r["tuberias"] + r["bombas"] + r["valvulas"]}
    code = {"closed": 0, "open": 1, "active": 2}
    for k, s in st.items():
        assert code[s] == e["status"][k], (k, s, e["status"][k])
    assert r["evidencia"]["residuo_masa_max_m3_s"] <= S.MASS_TOL_M3_S
    assert r["evidencia"]["verified"]


@pytest.mark.parametrize("name", ["net1_dw", "net1_dw_parallel", "net1_dw_cv110", "net1_dw_prv", "net1_dw_fcv"])
def test_net1_gap_is_only_the_friction_formula(monkeypatch, name):
    """With EPANET's two D-W choices swapped in (Swamee-Jain instead of Colebrook, and g = 32.2 ft/s² of its source
    code), the gap to EPANET closes from ~0.1 % to round-off: the solvers agree; only the friction details differed."""
    from fluids.friction import Swamee_Jain_1976
    monkeypatch.setattr(S, "_darcy_f", lambda Re, eD: 64.0 / Re if Re < 2000.0 else Swamee_Jain_1976(Re, eD))
    monkeypatch.setattr(S, "G", 32.2 * 0.3048)
    v = VAR[name]
    r = _run(v, system_curve=False)
    for k, Q in _flows(r).items():
        assert Q == pytest.approx(v["epanet"]["flow_m3_s"][k], abs=2e-7), k


def test_net1_prv_holds_its_setting_and_fcv_its_flow():
    r = _run(VAR["net1_dw_prv"])
    n21 = next(n for n in r["nodos"] if n["id"] == "21")
    assert n21["presion_Pa_g"] == pytest.approx(81.0 * 1000.0 * S.G, rel=1e-9)
    r = _run(VAR["net1_dw_fcv"])
    assert r["valvulas"][0]["caudal_m3_s"] == pytest.approx(0.004, abs=1e-12)


def test_net1_pump_station_system_curve_crosses_at_the_duty_point():
    r = _run(VAR["net1_dw_parallel"])
    st = r["estaciones_bombeo"][0]
    assert st["arreglo"] == "paralelo" and st["bombas_operando"] == 2
    assert st["cruce_curva_sistema_m"] < 1e-6
    req = [p["carga_requerida_m"] for p in st["curva_sistema"]]
    assert all(b > a for a, b in zip(req, req[1:]))                 # system curve rises with flow
    # the combined curve at the station head gives the station flow
    cc = st["curva_combinada"]
    hs = [c["carga_m"] for c in cc][::-1]
    qs = [c["caudal_total_m3_s"] for c in cc][::-1]
    import numpy as np
    assert float(np.interp(st["carga_m"], hs, qs)) == pytest.approx(st["caudal_total_m3_s"], rel=0.03)


def test_net1_hardy_cross_agrees_with_gradient():
    for name in ("net1_hw", "net1_dw", "net1_dw_parallel", "net1_dw_cv110"):
        hc = _run(VAR[name])["evidencia"]["hardy_cross"]
        assert hc["aplica"] and hc["convergio"], name
        assert hc["max_dif_caudal_m3_s"] < 1e-8 and hc["max_dif_carga_m"] < 1e-6, (name, hc)
    hc = _run(VAR["net1_dw_prv"])["evidencia"]["hardy_cross"]
    assert hc["aplica"] is False


# ---------------------------------------------------------------------------------------------- single pipe
@pytest.mark.parametrize("Q,K,dz", [(0.02, 0.0, 0.0), (0.02, 4.5, 12.0), (2e-5, 0.0, -3.0)])
def test_single_pipe_limit_equals_pipe_hydraulics(Q, K, dz):
    """Fixed pressure → one pipe → junction with the demand: the pressure drop is exactly pipe_dp's."""
    rho, mu, D, L, eps, P1 = 998.2, 1.002e-3, 0.1023, 250.0, 4.57e-5, 5e5
    nodes = [{"id": "in", "type": "pressure", "elevation_m": 0.0, "pressure_Pa_g": P1},
             {"id": "out", "type": "junction", "elevation_m": dz, "demand_m3_s": Q}]
    links = [{"id": "p", "type": "pipe", "from": "in", "to": "out", "length_m": L, "diameter_m": D,
              "roughness_m": eps, "minor_loss_K": K}]
    r = pipe_network(nodes, links, {"density_kg_m3": rho, "viscosity_Pa_s": mu})
    assert r["ok"], r
    ref = HY.pipe_dp(rho, mu, Q * rho, D, L, eps, dz, K)
    p2 = next(n for n in r["nodos"] if n["id"] == "out")["presion_Pa_g"]
    assert P1 - p2 == pytest.approx(ref["dP_total_Pa"], rel=1e-10)
    t = r["tuberias"][0]
    assert t["f_darcy"] == pytest.approx(ref["friction_factor_darcy"], rel=1e-10)   # Re via μ/ρ vs ρvD/μ: round-off
    assert t["Re"] == pytest.approx(ref["Re"], rel=1e-10) and t["regimen"] == ref["regime"]
    assert r["evidencia"]["friccion_vs_pipe_hydraulics_dif_rel_max"] < 1e-12


# ---------------------------------------------------------------------------------------------- pumps
def _pump_system(arrangement: str, h0=40.0, rr=2000.0, Hs=15.0, K=20.0, D=0.15):
    """Reservoir (0 m) → pumps → node → resistance K → reservoir at Hs. System: H = Hs + k·Q², k = K/(2gA²)."""
    nodes = [{"id": "R1", "type": "reservoir", "head_m": 0.0},
             {"id": "S", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.0},
             {"id": "R2", "type": "reservoir", "head_m": Hs}]
    pump = {"type": "pump", "curve": {"h0_m": h0, "r": rr, "n": 2.0}}
    if arrangement == "parallel":
        links = [{**pump, "id": "P1", "from": "R1", "to": "S"}, {**pump, "id": "P2", "from": "R1", "to": "S"}]
    else:
        nodes.append({"id": "M", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.0})
        links = [{**pump, "id": "P1", "from": "R1", "to": "M"}, {**pump, "id": "P2", "from": "M", "to": "S"}]
    links.append({"id": "V", "type": "pipe", "from": "S", "to": "R2", "length_m": 0.0, "diameter_m": D,
                  "roughness_m": 0.0, "minor_loss_K": K})
    k = K / (2 * S.G * (math.pi * D * D / 4) ** 2)
    return nodes, links, k


def test_parallel_pumps_closed_form():
    nodes, links, k = _pump_system("parallel")
    r = pipe_network(nodes, links, WATER)
    assert r["ok"], r
    Q = math.sqrt((40.0 - 15.0) / (k + 2000.0 / 4.0))               # h0 − r(Q/2)² = Hs + kQ²
    st = r["estaciones_bombeo"][0]
    assert st["caudal_total_m3_s"] == pytest.approx(Q, rel=1e-9)
    assert r["bombas"][0]["caudal_m3_s"] == pytest.approx(Q / 2, rel=1e-9)
    assert st["carga_m"] == pytest.approx(15.0 + k * Q * Q, rel=1e-9)
    single = math.sqrt((40.0 - 15.0) / (k + 2000.0))
    assert Q > single                                                # parallel gives more flow, not double
    assert st["cruce_curva_sistema_m"] < 1e-8
    for p in st["curva_sistema"]:
        assert p["carga_requerida_m"] == pytest.approx(15.0 + k * p["caudal_m3_s"] ** 2, rel=1e-9, abs=1e-9)


def test_series_pumps_closed_form_and_power():
    nodes, links, k = _pump_system("series")
    links[0]["efficiency"] = 0.7
    r = pipe_network(nodes, links, WATER)
    assert r["ok"], r
    Q = math.sqrt((80.0 - 15.0) / (k + 4000.0))                      # 2(h0 − rQ²) = Hs + kQ²
    assert r["bombas"][0]["caudal_m3_s"] == pytest.approx(Q, rel=1e-9)
    h1 = 40.0 - 2000.0 * Q * Q
    assert r["bombas"][0]["carga_m"] == pytest.approx(h1, rel=1e-9)
    assert r["bombas"][0]["potencia_hidraulica_kW"] == pytest.approx(998.2 * S.G * Q * h1 / 1e3, rel=1e-9)
    assert r["bombas"][0]["potencia_eje_kW"] == pytest.approx(998.2 * S.G * Q * h1 / 1e3 / 0.7, rel=1e-9)
    assert r["bombas"][1]["potencia_eje_kW"] is None                 # no efficiency given: never assumed
    assert r["bombas_en_serie"] == [(["P1"], ["P2"])]


def test_pump_cannot_deliver_closes_like_a_check():
    nodes, links, _ = _pump_system("parallel", Hs=45.0)               # static head above shut-off
    r = pipe_network(nodes, links, WATER)
    assert r["ok"], r
    assert all(p["estado"] == "closed" and p["caudal_m3_s"] == 0.0 for p in r["bombas"])


def test_npsh_available_by_hand():
    """Open tank 3 m below the pump, 10 m suction line: NPSHa = Patm/ρg − Pv/ρg − 3 − h_L (Çengel's form)."""
    rho, mu, Pv, Patm, D, L, eps = 998.2, 1.002e-3, 2339.0, 101325.0, 0.1023, 10.0, 4.57e-5
    Q = 0.015
    nodes = [{"id": "T", "type": "reservoir", "head_m": 0.0},
             {"id": "s", "type": "junction", "elevation_m": 3.0, "demand_m3_s": 0.0},
             {"id": "d", "type": "junction", "elevation_m": 3.0, "demand_m3_s": Q},
             {"id": "R", "type": "reservoir", "head_m": 20.0}]
    links = [{"id": "suc", "type": "pipe", "from": "T", "to": "s", "length_m": L, "diameter_m": D, "roughness_m": eps},
             {"id": "B", "type": "pump", "from": "s", "to": "d", "curve": {"h0_m": 50.0, "r": 1e5, "n": 2.0},
              "npshr_points": [[0.0, 1.5], [0.03, 4.0]]},
             {"id": "fcv", "type": "fcv", "from": "d", "to": "R", "diameter_m": D, "setting_flow_m3_s": 0.0}]
    r = pipe_network(nodes, links, {"density_kg_m3": rho, "viscosity_Pa_s": mu, "vapor_pressure_Pa": Pv},
                     atmospheric_pressure_Pa=Patm)
    assert r["ok"], r
    hL = HY.pipe_dp(rho, mu, Q * rho, D, L, eps)["dP_friction_Pa"] / (rho * S.G)
    npsh = r["bombas"][0]["npsh"]
    assert npsh["NPSHa_m"] == pytest.approx((Patm - Pv) / (rho * S.G) - 3.0 - hL, rel=1e-10)
    assert npsh["NPSHr_m"] == pytest.approx(1.5 + 2.5 * Q / 0.03, rel=1e-12)
    assert npsh["cavita"] is False and npsh["margen_m"] > 0
    r2 = pipe_network(nodes, links, {"density_kg_m3": rho, "viscosity_Pa_s": mu})
    assert "faltan datos" in r2["bombas"][0]["npsh"]["nota"]          # no Patm / Pv: not computed, never assumed


# ---------------------------------------------------------------------------------------------- checks and data
def test_check_valve_blocks_reverse_flow():
    nodes = [{"id": "A", "type": "reservoir", "head_m": 10.0}, {"id": "B", "type": "reservoir", "head_m": 30.0},
             {"id": "j", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.01}]
    pipe = {"type": "pipe", "length_m": 100.0, "diameter_m": 0.1, "roughness_m": 4.57e-5}
    links = [{**pipe, "id": "a", "from": "A", "to": "j"}, {**pipe, "id": "b", "from": "B", "to": "j"},
             {**pipe, "id": "cv", "from": "j", "to": "A", "check_valve": True}]
    r = pipe_network(nodes, links, WATER)
    assert r["ok"], r
    cv = next(p for p in r["tuberias"] if p["id"] == "cv")
    assert cv["estado"] == "open" and cv["caudal_m3_s"] > 0            # B is higher: flow j → A allowed
    links[2] = {**links[2], "from": "A", "to": "j"}                    # now the check only allows A → j
    r = pipe_network(nodes, links, WATER)
    cv = next(p for p in r["tuberias"] if p["id"] == "cv")
    assert cv["estado"] == "closed" and cv["caudal_m3_s"] == 0.0
    assert r["evidencia"]["residuo_masa_max_m3_s"] <= S.MASS_TOL_M3_S


def test_user_data_is_never_defaulted():
    nodes = [{"id": "A", "type": "reservoir", "head_m": 10.0}, {"id": "j", "type": "junction", "demand_m3_s": 0.01}]
    links = [{"id": "p", "type": "pipe", "from": "A", "to": "j", "length_m": 10.0, "diameter_m": 0.1,
              "roughness_m": 1e-4}]
    r = pipe_network(nodes, links, WATER)
    assert not r["ok"] and "elevation_m" in r["error"]
    nodes[1]["elevation_m"] = 0.0
    r = pipe_network(nodes, [{**links[0], "roughness_m": None}], WATER)
    assert not r["ok"] and "roughness_m o material" in r["error"]
    r = pipe_network(nodes, [{**links[0], "roughness_m": None, "material": "Cast iron"}], WATER)
    assert r["ok"] and any("Rugosidad" in s and "fluids" in s for s in r["supuestos"])
    r = pipe_network(nodes, [{"id": "b", "type": "pump", "from": "A", "to": "j"}], WATER)
    assert not r["ok"] and "curva" in r["error"]
    r = pipe_network(nodes, links, {"density_kg_m3": 998.0})
    assert not r["ok"] and "viscosity_Pa_s" in r["error"]


def test_fittings_and_material_and_nps():
    nodes = [{"id": "A", "type": "reservoir", "head_m": 30.0},
             {"id": "j", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.01}]
    links = [{"id": "p", "type": "pipe", "from": "A", "to": "j", "length_m": 50.0, "nps_in": 4, "schedule": "40",
              "material": "commercial steel", "fittings": [{"name": "Elbow, 90°, threaded, standard, (r/D = 1)",
                                                             "count": 2}]}]
    r = pipe_network(nodes, links, WATER)
    assert r["ok"], r
    t = r["tuberias"][0]
    assert t["K_accesorios"] > 0
    from fluids.piping import nearest_pipe
    Di = nearest_pipe(NPS=4, schedule="40")[1]
    ref = HY.pipe_dp(998.2, 1.002e-3, 0.01 * 998.2, Di, 50.0, 4.57e-5, 0.0, t["K_accesorios"])
    assert t["dP_friccion_Pa"] == pytest.approx(ref["dP_friction_Pa"] + ref["dP_fittings_Pa"], rel=1e-9)


def test_isolated_node_with_demand_is_an_error():
    nodes = [{"id": "A", "type": "reservoir", "head_m": 10.0},
             {"id": "j", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.01}]
    links = [{"id": "cv", "type": "pipe", "from": "j", "to": "A", "length_m": 10.0, "diameter_m": 0.1,
              "roughness_m": 1e-4, "check_valve": True}]
    r = pipe_network(nodes, links, WATER)
    assert not r["ok"] and "aislado" in r["error"]


# ---------------------------------------------------------------------------------------------- published examples
# Jeppson, R. W. (1974) "Steady Flow Analysis of Pipe Networks: An Instructional Manual", Utah Water Research
# Laboratory, Reports paper 300 (open access, digitalcommons.usu.edu/water_rep/300). English units; read from the page
# images (notes in the M13 side-session folder). Jeppson states water at 68 °F for A2; for A4 no temperature is
# printed, 68 °F is ASSUMED (flows and pump heads do not change between 50 and 68 °F at the printed precision).
FT, CFS, IN, GPM = 0.3048, 0.3048 ** 3, 0.0254, 6.30901964e-5
W68 = {"compound": "water", "temperature_K": 293.15}


def _jeppson_loops(pipes: dict, ext_gpm: dict, eps_in: float, eps_special: dict) -> dict:
    """Flow-only network (external flows, no reservoir): junction [1] becomes the fixed-head reference, so its computed
    supply must equal its printed external flow."""
    nodes = [{"id": "1", "type": "reservoir", "head_m": 100.0}]
    nodes += [{"id": str(j), "type": "junction", "elevation_m": 0.0, "demand_m3_s": -e * GPM}
              for j, e in ext_gpm.items() if j != 1]
    links = [{"id": str(i), "type": "pipe", "from": str(a), "to": str(b), "diameter_m": D * IN, "length_m": L * FT,
              "roughness_m": eps_special.get(i, eps_in) * IN} for i, (a, b, D, L) in pipes.items()]
    r = pipe_network(nodes, links, W68)
    assert r["ok"], r
    assert next(n for n in r["nodos"] if n["id"] == "1")["suministro_m3_s"] == pytest.approx(ext_gpm[1] * GPM, rel=1e-9)
    return r


def test_jeppson_p43_a_four_loops_darcy_weisbach():
    pipes = {1: (1, 2, 18, 1600), 2: (2, 3, 15, 2000), 3: (1, 3, 18, 2400), 4: (2, 4, 12, 1800), 5: (4, 3, 12, 1900),
             6: (5, 4, 10, 1300), 7: (5, 6, 15, 1700), 8: (3, 6, 18, 2000), 9: (5, 7, 24, 1200), 10: (7, 6, 15, 1800)}
    r = _jeppson_loops(pipes, {1: 2000, 2: -300, 4: -500, 5: 1500, 7: -1000, 3: -900, 6: -800}, .0102, {6: .009, 8: .009})
    printed = [2.19, 0.91, 2.26, 0.61, -0.27, 0.23, 0.65, -0.90, 2.47, 0.24]
    hf = [0.500, 0.297, 0.797, 0.389, 0.092, 0.110, 0.134, 0.116, 0.111, 0.023]
    t = {p["id"]: p for p in r["tuberias"]}
    for i, (Qp, hp) in enumerate(zip(printed, hf), start=1):
        Q = t[str(i)]["caudal_m3_s"] / CFS
        if i == 8:                       # drawn direction of pipe 8 is not legible in the scan: compare magnitude
            Q, Qp = abs(Q), abs(Qp)
        assert Q == pytest.approx(Qp, abs=0.006), i
        assert abs(t[str(i)]["perdida_carga_m"]) / FT == pytest.approx(hp, rel=0.02, abs=0.003), i
    assert r["evidencia"]["verified"] and r["evidencia"]["hardy_cross"]["lazos"] == 4


def test_jeppson_p43_b_five_loops_darcy_weisbach():
    pipes = {1: (1, 2, 24, 1900), 2: (2, 3, 18, 2000), 3: (4, 3, 24, 2000), 4: (1, 4, 18, 1400), 5: (2, 5, 24, 1500),
             6: (7, 5, 24, 1200), 7: (5, 6, 18, 2000), 8: (7, 3, 15, 3300), 9: (7, 8, 18, 1300), 10: (8, 6, 12, 1600),
             11: (8, 9, 18, 700), 12: (6, 9, 24, 1500), 13: (3, 6, 24, 1500)}
    r = _jeppson_loops(pipes, {1: 1500, 7: 2500, 5: 500, 4: -500, 3: -2000, 6: -500, 9: -1500, 2: 0, 8: 0},
                       .015, {10: .01, 11: .009})
    printed = [0.95, 2.03, 1.28, 2.39, -1.08, 2.05, 2.09, 1.06, 2.46, 0.39, 2.06, 1.28, -0.09]
    t = {p["id"]: p for p in r["tuberias"]}
    for i, Qp in enumerate(printed, start=1):
        assert t[str(i)]["caudal_m3_s"] / CFS == pytest.approx(Qp, abs=0.006), i
    assert r["evidencia"]["hardy_cross"]["lazos"] == 5 and r["evidencia"]["verified"]


def _jeppson_three_pumps(pumps: bool, resistance: dict | None = None) -> dict:
    """Jeppson pp. 46-49: reservoirs at 100, 105 and 95 ft feed a 1-loop network (2 cfs out at [2]) through pipes 1,
    5 and 7, each with the same pump h = −10.328Q² + 2.823Q + 22.289 (ft, cfs); junction elevations 0."""
    DL = {1: (8, 1000), 2: (6, 2000), 3: (8, 2000), 4: (8, 1000), 5: (6, 1000), 6: (6, 1000), 7: (6, 1000)}
    nodes = [{"id": "R1", "type": "reservoir", "head_m": 100 * FT}, {"id": "R2", "type": "reservoir", "head_m": 105 * FT},
             {"id": "R3", "type": "reservoir", "head_m": 95 * FT}]
    nodes += [{"id": str(j), "type": "junction", "elevation_m": 0.0, "demand_m3_s": 2.0 * CFS if j == 2 else 0.0}
              for j in (1, 2, 3, 4)]
    curve = [[q * CFS, (-10.328 * q * q + 2.823 * q + 22.289) * FT] for q in (0.0, 0.5, 1.0, 1.5)]
    links = []

    def pipe(i, a, b):
        if resistance:
            K, n = resistance[i]
            return {"id": str(i), "type": "pipe", "from": a, "to": b, "resistance": {"K": K * FT / CFS ** n, "n": n}}
        return {"id": str(i), "type": "pipe", "from": a, "to": b, "diameter_m": DL[i][0] * IN,
                "length_m": DL[i][1] * FT, "roughness_m": 0.012 * IN}
    for i, (a, b) in {1: ("R1", "1"), 5: ("R2", "3"), 7: ("R3", "4")}.items():
        if pumps:
            nodes.append({"id": f"p{i}", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.0})
            links.append({"id": f"P{i}", "type": "pump", "from": a, "to": f"p{i}", "curve_points": curve})
            a = f"p{i}"
        links.append(pipe(i, a, b))
    for i, (a, b) in {2: ("1", "2"), 3: ("3", "2"), 4: ("4", "3"), 6: ("1", "4")}.items():
        links.append(pipe(i, a, b))
    r = pipe_network(nodes, links, W68)
    assert r["ok"], r
    return r


def test_jeppson_p47_three_pumps_three_reservoirs():
    """Printed solution p. 47: flows to 3 decimals, pump heads 17.36/18.46/21.78 ft, junction heads to 0.1 ft.
    The K's printed with the problem (6.86, 55.7 … for n = 1.85) do not reproduce its own head losses; the printed
    answer is the Darcy-Weisbach solution with e = 0.012 in stated in the problem, which is what is solved here."""
    r = _jeppson_three_pumps(pumps=True)
    t = {p["id"]: p for p in r["tuberias"]}
    for i, Qp in enumerate([0.841, 0.666, 1.334, 0.573, 0.761, 0.175, 0.398], start=1):
        assert t[str(i)]["caudal_m3_s"] / CFS == pytest.approx(Qp, abs=0.0015), i
    for i, hp in enumerate([3.15, 17.63, 15.46, 1.50, 11.42, 0.674, 3.24], start=1):
        assert t[str(i)]["perdida_carga_m"] / FT == pytest.approx(hp, rel=0.015), i
    heads = {n["id"]: n["carga_m"] / FT for n in r["nodos"]}
    for j, Hp in {"1": 114.2, "2": 96.6, "3": 112.0, "4": 113.5}.items():
        assert heads[j] == pytest.approx(Hp, abs=0.15), j
    for p, hp in zip(r["bombas"], [17.36, 18.46, 21.78]):
        assert p["carga_m"] / FT == pytest.approx(hp, abs=0.015), p["id"]
    assert r["evidencia"]["verified"]


def test_jeppson_p49_three_reservoirs_hazen_williams_K():
    """Problem 5 (p. 49): no pumps, h = K·Q^1.852 with the printed K's (C = 120): flows and head losses."""
    K = {1: 4.85, 2: 39.3, 3: 9.69, 4: 4.85, 5: 19.7, 6: 19.7, 7: 19.7}
    r = _jeppson_three_pumps(pumps=False, resistance={i: (k, 1.852) for i, k in K.items()})
    t = {p["id"]: p for p in r["tuberias"]}
    for i, (Qp, hp) in enumerate(zip([0.965, 0.684, 1.316, 0.522, 0.795, 0.281, 0.241],
                                     [4.53, 19.45, 16.13, 1.45, 12.86, 1.87, 1.41]), start=1):
        assert t[str(i)]["caudal_m3_s"] / CFS == pytest.approx(Qp, abs=0.0015), i
        assert t[str(i)]["perdida_carga_m"] / FT == pytest.approx(hp, rel=0.01, abs=0.02), i
    assert next(n for n in r["nodos"] if n["id"] == "2")["carga_m"] / FT == pytest.approx(76.02, abs=0.05)


def test_larock_example_2_7_three_reservoirs():
    """Larock, Jeppson & Watters (2000) Hydraulics of Pipeline Systems, Ex. 2.7 (as reprinted in the Pipe Flow Expert
    verification document): 0.1023 / 0.0200 / 0.0622 m³/s with 0.06 m³/s drawn at the junction. Water 10 °C."""
    nodes = [{"id": "A", "type": "reservoir", "head_m": 100.0}, {"id": "B", "type": "reservoir", "head_m": 85.0},
             {"id": "C", "type": "reservoir", "head_m": 60.0},
             {"id": "J", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.06}]   # elevation not printed: flows do not need it
    pipe = {"type": "pipe", "roughness_m": 5e-4}
    links = [{**pipe, "id": "AJ", "from": "A", "to": "J", "length_m": 2000.0, "diameter_m": 0.30},
             {**pipe, "id": "BJ", "from": "B", "to": "J", "length_m": 1500.0, "diameter_m": 0.25},
             {**pipe, "id": "JC", "from": "J", "to": "C", "length_m": 3000.0, "diameter_m": 0.25}]
    r = pipe_network(nodes, links, {"compound": "water", "temperature_K": 283.15})
    assert r["ok"], r
    q = {p["id"]: p["caudal_m3_s"] for p in r["tuberias"]}
    assert q["AJ"] == pytest.approx(0.1023, abs=2e-4)
    assert q["BJ"] == pytest.approx(0.0200, abs=1e-4)
    assert q["JC"] == pytest.approx(0.0622, abs=1e-4)


def test_evett_liu_12_11_three_reservoirs_loose():
    """Evett & Liu (1989) 2500 Solved Problems, 12.11: the book stops after three trial junction heads, so its
    answer (77.8 / 46.2 / 31.7 L/s, HGL 19.7 m) is ~1-1.5 % from converged; water 10 °C is the reprint's assumption."""
    nodes = [{"id": "L", "type": "reservoir", "head_m": 15.0}, {"id": "M", "type": "reservoir", "head_m": 25.0},
             {"id": "H", "type": "reservoir", "head_m": 28.0},
             {"id": "J", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.0}]
    links = [{"id": "1", "type": "pipe", "from": "J", "to": "L", "length_m": 600.0, "diameter_m": 0.3, "roughness_m": 3e-3},
             {"id": "2", "type": "pipe", "from": "M", "to": "J", "length_m": 300.0, "diameter_m": 0.2, "roughness_m": 1e-3},
             {"id": "3", "type": "pipe", "from": "H", "to": "J", "length_m": 1000.0, "diameter_m": 0.2, "roughness_m": 1e-3}]
    r = pipe_network(nodes, links, {"compound": "water", "temperature_K": 283.15})
    q = {p["id"]: p["caudal_m3_s"] * 1e3 for p in r["tuberias"]}
    for k, v in {"1": 77.8, "2": 46.2, "3": 31.7}.items():
        assert q[k] == pytest.approx(v, rel=0.02), k
    assert next(n for n in r["nodos"] if n["id"] == "J")["carga_m"] == pytest.approx(19.7, abs=0.2)


PSI = 6894.757293168


@pytest.mark.parametrize("case", ["ex1", "ex2", "ex3", "ex4"])
def test_cameron_npsh_examples(case):
    """Cameron Hydraulic Data, 18th ed. (1994), pp. 1-13 to 1-15 (as reprinted in the Pipe Flow Expert verification
    document): source 10 ft below/above the pump, 2.92 ft of suction friction. Printed NPSHa 20.26 / 40.26 / 7.08 /
    7.08 ft; suction lift 12.92 ft (ex1), suction head 7.08 ft (ex2) and 317.69 ft (ex4)."""
    Q = 0.01
    z_src = {"ex1": -10.0, "ex2": 10.0, "ex3": 10.0, "ex4": 10.0}[case] * FT
    if case in ("ex1", "ex2"):
        fluid, Patm = W68, 14.696 * PSI
        src = {"id": "T", "type": "reservoir", "head_m": z_src}
    elif case == "ex3":                          # boiling water: surface pressure = vapour pressure
        fluid, Patm = {"density_kg_m3": 958.4, "viscosity_Pa_s": 2.8e-4, "vapor_pressure_Pa": 14.696 * PSI}, 14.696 * PSI
        src = {"id": "T", "type": "reservoir", "head_m": z_src}
    else:                                        # 350 °F water, SG 0.8904, Pv 134.60 psia, vessel at 119.90 psig
        fluid = {"density_kg_m3": 890.4, "viscosity_Pa_s": 1.5e-4, "vapor_pressure_Pa": 134.60 * PSI}
        Patm = 14.696 * PSI
        src = {"id": "T", "type": "pressure", "elevation_m": z_src, "pressure_Pa_g": 119.90 * PSI}
    nodes = [src, {"id": "s", "type": "junction", "elevation_m": 0.0, "demand_m3_s": 0.0},
             {"id": "d", "type": "junction", "elevation_m": 0.0, "demand_m3_s": Q}]
    links = [{"id": "suc", "type": "pipe", "from": "T", "to": "s", "resistance": {"K": 2.92 * FT / Q ** 2, "n": 2.0}},
             {"id": "B", "type": "pump", "from": "s", "to": "d", "curve": {"h0_m": 400.0, "r": 1e5, "n": 2.0}}]
    r = pipe_network(nodes, links, fluid, atmospheric_pressure_Pa=Patm)
    assert r["ok"], r
    n = r["bombas"][0]["npsh"]
    printed = {"ex1": 20.26, "ex2": 40.26, "ex3": 7.08, "ex4": 7.08}[case]
    assert n["NPSHa_m"] / FT == pytest.approx(printed, abs=0.015)
    suction = {"ex1": -12.92, "ex2": 7.08, "ex4": 317.69}.get(case)
    if suction is not None:
        assert n["carga_succion_manometrica_m"] / FT == pytest.approx(suction, abs=0.015)
