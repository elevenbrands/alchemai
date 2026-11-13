"""Pinch from a DWSIM flowsheet (bloque M5b): stream extraction, CP segmentation and the end-to-end Kemp case.

Kemp, *Pinch Analysis and Process Integration* (2007), four streams, ΔTmin = 10 K: Q_H,min = 20 kW, Q_C,min = 60 kW,
pinch 90/80 °C (same numbers as tests/test_pinch.py). Here the four streams are liquid water at 20 bar in a real
flowsheet: one exchanger E-1 between streams 2 and 3 followed by a cooler and a heater, plus a heater on stream 1 and
a cooler on stream 4. Flows are calibrated in DWSIM so each stream's duty equals Kemp's CP·ΔT; with one linear CP per
stream the tool must give Kemp's targets, and with the real (segmented) water cp it gives slightly different ones.
Condensing steam is checked against IAPWS: h_fg at 1 atm = 2256.4 kJ/kg."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.energy import stream_extraction as X  # noqa: E402

K = 273.15


def _piece(T_in, T_out, H, name="a", out="b", kind="heater", unit="U"):
    return X.Piece(unit=unit, kind=kind, side="", inlet=name, outlet=out, T_in=T_in, T_out=T_out, P_in=1e5, P_out=1e5,
                   H_in=H(T_in), H_out=H(T_out), duty_reported_W=abs(H(T_out) - H(T_in)))


def _fake_flash(Hs):
    return lambda reqs: [Hs[i](T) for i, T, _ in reqs]


def test_constant_cp_gives_one_segment():
    H = lambda T: 2000.0 * T                                                     # noqa: E731
    p = _piece(300.0, 400.0, H)
    X.sample([p], _fake_flash([H]), 1.0)
    assert X.segments(p, 1.0) == [(300.0, 400.0, pytest.approx(2e5))]


def test_phase_change_step_is_located_and_closes():
    Tsat, lat = 373.0, 2.0e6
    H = lambda T: 1000.0 * T + (lat if T > Tsat else 0.0)                         # noqa: E731
    p = _piece(450.0, 300.0, H, kind="cooler")                                    # condenser: cools through Tsat
    X.sample([p], _fake_flash([H]), 1.0)
    segs = X.segments(p, 1.0)
    big = max(segs, key=lambda s: s[2])
    assert big[0] >= Tsat >= big[1] and big[0] - big[1] <= X.MIN_SPAN_K
    assert big[2] == pytest.approx(lat, rel=0.01)
    assert sum(s[2] for s in segs) == pytest.approx(p.Q, rel=1e-12)


def test_isothermal_piece_and_chaining():
    H1 = lambda T: 3000.0 * T                                                     # noqa: E731
    a = _piece(450.0, 400.0, H1, name="s", out="m", kind="heat_exchanger", unit="E")
    b = _piece(400.0, 350.0, H1, name="m", out="o", kind="cooler", unit="C")
    iso = X.Piece(unit="R", kind="heater", side="", inlet="x", outlet="y", T_in=373.0, T_out=373.0, P_in=1e5,
                  P_out=1e5, H_in=0.0, H_out=5e5, duty_reported_W=5e5)
    X.sample([a, b, iso], _fake_flash([H1, H1, None]), 1.0)
    tab = X.stream_table([a, b, iso], 1.0)
    rows = {r["name"]: r for r in tab["rows"]}
    assert set(rows) == {"H1: s→o", "C2: x→y"}
    assert rows["H1: s→o"]["duty_W"] == pytest.approx(3e5) and rows["H1: s→o"]["units"] == ["E", "C"]
    (seg,) = rows["C2: x→y"]["segments"]
    assert (seg["T_supply_K"], seg["T_target_K"]) == (372.5, 373.5)               # latent over a 1 K span
    assert all(c["ok"] and c["segments_close"] for c in tab["checks"])


# ----------------------------------------------------------------------------- DWSIM end to end
KEMP = [("1", 20, 135, 2e3), ("2", 170, 60, 3e3), ("3", 80, 140, 4e3), ("4", 150, 30, 1.5e3)]
P_BAR20 = 20e5


@pytest.fixture(scope="module")
def dwsim():
    from packages.dwsim_client import tools as T
    from packages.dwsim_client import tools_energy as TE
    return T, TE


def _kemp_flowsheet(T):
    # calibration: water duty per kg/s for each Kemp stream, so m = CP·ΔT / Δh
    fid = T.new_flowsheet("cal", ["Water"], "SteamTables")["flowsheet_id"]
    for n, a, b, _ in KEMP:
        T.add_material_stream(fid, f"i{n}", temperature_K=a + K, pressure_Pa=P_BAR20, mass_flow_kg_s=1.0,
                              mole_fractions={"Water": 1.0})
        T.add_material_stream(fid, f"o{n}")
        T.add_unit(fid, "heater" if b > a else "cooler", f"U{n}", {"outlet_T_K": b + K})
        T.connect(fid, f"U{n}", feeds=[f"i{n}"], products=[f"o{n}"])
    assert T.solve(fid)["ok"]
    st = T.SESSIONS[fid]["report"].streams
    m = {n: cp * abs(b - a) / abs(st[f"o{n}"].enthalpy_kw - st[f"i{n}"].enthalpy_kw) / 1e3 for n, a, b, cp in KEMP}
    T.close_flowsheet(fid)

    fid = T.new_flowsheet("kemp", ["Water"], "SteamTables")["flowsheet_id"]
    for n, Tc in (("1", 20), ("2", 170), ("3", 80), ("4", 150)):
        T.add_material_stream(fid, f"s{n}", temperature_K=Tc + K, pressure_Pa=P_BAR20, mass_flow_kg_s=m[n],
                              mole_fractions={"Water": 1.0})
    for n in ("s1o", "s2a", "s2o", "s3a", "s3o", "s4o"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "heat_exchanger", "E-1", {"UA_W_K": 3000})
    T.connect(fid, "E-1", feeds=["s2", "s3"], products=["s2a", "s3a"])
    for u, kind, Tt, a, b in (("H-1", "heater", 135, "s1", "s1o"), ("C-2", "cooler", 60, "s2a", "s2o"),
                              ("H-3", "heater", 140, "s3a", "s3o"), ("C-4", "cooler", 30, "s4", "s4o")):
        T.add_unit(fid, kind, u, {"outlet_T_K": Tt + K})
        T.connect(fid, u, feeds=[a], products=[b])
    return fid


def test_kemp_flowsheet_end_to_end(dwsim):
    T, TE = dwsim
    fid = _kemp_flowsheet(T)
    r = TE.pinch_from_flowsheet(fid, 10.0, dT_min_sweep_K=[10, 20])
    ev = r["evidence"]
    assert ev["verified"] and ev["unit_balances_ok"] and ev["exchangers_balanced"] and ev["global_balance_closed"]
    assert all(c["rel_err"] <= 1e-6 for c in r["unit_balance"])                  # ΔH of the streams = DWSIM duty
    # four process streams: the exchanger sides are chained with their cooler/heater
    rows = {r_["name"]: r_ for r_ in r["extracted_streams"]}
    assert len(rows) == 4 and rows["H1: s2→s2o"]["units"] == ["E-1(hot)", "C-2"]
    assert rows["H1: s2→s2o"]["CP_linear_W_K"] == pytest.approx(3e3, rel=1e-6)
    # one linear CP per stream = Kemp's book targets
    lin = r["linear_cp_targets"]
    assert lin["Q_hot_min_W"] == pytest.approx(20e3, abs=5.0) and lin["Q_cold_min_W"] == pytest.approx(60e3, abs=5.0)
    (p,) = lin["pinch"]
    assert (p["T_hot_side_K"] - K, p["T_cold_side_K"] - K) == pytest.approx((90.0, 80.0))
    # real water cp (segmented): same pinch, targets within 5 % of the linear ones, and minimum units on real streams
    assert r["Q_hot_min_W"] == pytest.approx(20e3, rel=0.05) and r["Q_hot_min_W"] != pytest.approx(20e3, abs=100)
    assert r["pinch"][0]["T_hot_side_K"] - K == pytest.approx(90.0)
    assert r["minimum_units"] == {"above_pinch": 4, "below_pinch": 3, "total_MER": 7, "without_pinch_split": 5}
    # current utilities: Σ heaters / Σ coolers; saving consistent on both sides
    cu = r["current_utilities"]
    assert cu["hot_W"] == pytest.approx(470e3 - cu["recovered_in_exchangers_W"]["E-1"], rel=1e-6)
    assert r["current_vs_target"]["balance_consistent"]
    assert [s["dT_min_K"] for s in r["dT_min_sweep"]] == [10.0, 20.0]
    T.close_flowsheet(fid)


def test_condensing_steam_latent_heat_matches_iapws(dwsim):
    T, TE = dwsim
    fid = T.new_flowsheet("cond", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "v", temperature_K=150 + K, pressure_Pa=101325, mass_flow_kg_s=0.1, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "w", temperature_K=20 + K, pressure_Pa=5e5, mass_flow_kg_s=0.6, mole_fractions={"Water": 1.0})
    for n in ("vo", "wo"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "cooler", "C-1", {"outlet_T_K": 50 + K})
    T.connect(fid, "C-1", feeds=["v"], products=["vo"])
    T.add_unit(fid, "heater", "H-1", {"outlet_T_K": 140 + K})
    T.connect(fid, "H-1", feeds=["w"], products=["wo"])
    r = TE.pinch_from_flowsheet(fid, 10.0)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    hot = next(x for x in r["extracted_streams"] if x["type"] == "hot")
    lat = max(hot["segments"], key=lambda s: s["duty_W"])
    assert lat["T_supply_K"] - lat["T_target_K"] <= X.MIN_SPAN_K
    assert lat["T_target_K"] <= 99.97 + K <= lat["T_supply_K"] + 0.01
    assert lat["duty_W"] == pytest.approx(0.1 * 2256.4e3, rel=0.005)
    # the linear-CP shortcut spreads the latent heat over 100 K and gets the targets wrong
    assert abs(r["Q_hot_min_W"] - r["linear_cp_targets"]["Q_hot_min_W"]) > 10e3
    T.close_flowsheet(fid)


def test_unsolved_or_no_pair_is_refused(dwsim):
    T, TE = dwsim
    fid = T.new_flowsheet("one", ["Water"], "SteamTables")["flowsheet_id"]
    T.add_material_stream(fid, "a", temperature_K=300, pressure_Pa=2e5, mass_flow_kg_s=1.0, mole_fractions={"Water": 1.0})
    T.add_material_stream(fid, "b")
    T.add_unit(fid, "heater", "H", {"outlet_T_K": 350})
    T.connect(fid, "H", feeds=["a"], products=["b"])
    with pytest.raises(ValueError, match="calientes y frías"):
        TE.pinch_from_flowsheet(fid, 10.0)
    T.close_flowsheet(fid)
