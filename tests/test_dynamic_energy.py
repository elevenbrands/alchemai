"""Validation of the energy mode of the dynamic flash drum and of the dynamic binary column (block L2).

References and what is compared:
  [KAN] Kantor, CBE 20255 notebook 07.09 (read 2026-11-14, also used by tests/test_dynamic_flowsheet.py),
        https://github.com/jckantor/CBE20255/blob/master/notebooks/07.09-Isothermal-Flash-and-the-Rachford-Rice-Equation.ipynb
        Antoine constants of acetone (7.02447, 1161.0, 224) and ethanol (8.04494, 1554.3, 222.65), mmHg and °C, converted
        here to log10(P/Pa) = A − B/(T/K + C). They are the only literature numbers used: the heat capacities and heats of
        vaporisation below are ILLUSTRATIVE inputs (the comparisons are solver against independent solver / closed form
        on the same physics, so they do not need to be the true values of acetone and ethanol).
  [THERMO] thermo 0.6.1 (Caleb Bell) FlashVL with IdealGas + GibbsExcessLiquid (Raoult, caloric_basis 'Hvap') built with
        the same Antoine, constant ideal-gas cp and ΔHvap(T) = ΔH_ref + (cp_V − cp_L)(T − T_ref): an independent P-H flash
        and TP flash implementation of the same thermodynamics.
  [SKA] Skogestad column A (see tests/test_dynamic_flowsheet.py): the existing constant-molar-overflow model and profile.
  Analytical: pure-component adiabatic flash (closed form T and P), U(t) = U0 + Q·t of a closed heated drum, own
  steady MESH solve of the column (written below, independent of packages.dynamics.column).
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq, root

from packages.dwsim_client import tools_dynamic_flowsheet as TDF
from packages.dynamics import flash_drum as FD
from packages.dynamics import flowsheet as FS

sys.path.insert(0, str(Path(__file__).parent))
from test_dynamic_flowsheet import COLA_X, KMIN, STAGES, _cola  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

MM = math.log10(101325.0 / 760.0)                         # mmHg → Pa in log10
ANT = {"acetone": (7.02447 + MM, 1161.0, 224.0 - 273.15), "ethanol": (8.04494 + MM, 1554.3, 222.65 - 273.15)}   # [KAN]
TH = {"acetone": (125.0, 75.0, 30000.0), "ethanol": (112.0, 65.0, 40000.0)}   # cp_L, cp_V J/mol/K, ΔH at T_REF: illustrative
T_REF = 330.0
COMP = ["acetone", "ethanol"]


def ok(r):
    json.dumps(r)
    assert r["evidence"]["verified"], r["evidence"]["reasons"]
    return r


def psat(c, T):
    A, B, C = ANT[c]
    return 10.0 ** (A - B / (T + C))


# ------------------------------------------------------------------------------------------------------ thermo package
def _thermo_flasher(rhom):
    from thermo import (ChemicalConstantsPackage, EnthalpyVaporization, FlashVL, GibbsExcessLiquid, HeatCapacityGas,
                        IdealGas, PropertyCorrelationsPackage, VaporPressure, VolumeLiquid)
    vps, cps, hvs, vls = [], [], [], []
    for c in COMP:
        (A, B, C), (cl, cv, dh) = ANT[c], TH[c]
        vp = VaporPressure()
        vp.add_correlation("a", "Antoine", 200, 600, A=A, B=B, C=C, base=10.0)
        cg = HeatCapacityGas()
        cg.add_correlation("c", "constant", 200, 600, value=cv)
        hv = EnthalpyVaporization()
        hv.add_correlation("h", "linear", 200, 600, A=dh - (cv - cl) * T_REF, B=cv - cl)
        vl = VolumeLiquid()
        vl.add_correlation("v", "constant", 200, 600, value=1.0 / rhom)
        vps.append(vp), cps.append(cg), hvs.append(hv), vls.append(vl)
    consts = ChemicalConstantsPackage(Tcs=[508.1, 514.0], Pcs=[4.7e6, 6.1e6], omegas=[0.307, 0.644], MWs=[58.08, 46.07],
                                      names=COMP)            # only used by thermo for initial guesses
    props = PropertyCorrelationsPackage(constants=consts, VaporPressures=vps, HeatCapacityGases=cps,
                                        EnthalpyVaporizations=hvs, VolumeLiquids=vls, skip_missing=True)
    gas = IdealGas(HeatCapacityGases=cps, T=300.0, P=1e5, zs=[0.5, 0.5])
    liq = GibbsExcessLiquid(VaporPressures=vps, VolumeLiquids=vls, HeatCapacityGases=cps, EnthalpyVaporizations=hvs,
                            equilibrium_basis="Psat", caloric_basis="Hvap", T=300.0, P=1e5, zs=[0.5, 0.5])
    return FlashVL(consts, props, liquid=liq, gas=gas)


# ------------------------------------------------------------------------------------------------------ flash drum
def _drum(Q=0.0, TF=360.0, z=(0.6, 0.4), vap=None, liq=None, F=10.0, z0=None, T0=342.0, h0=0.8, comps=COMP):
    zz = dict(zip(comps, z))
    return {"components": list(comps), "volume_m3": 0.6, "area_m2": 0.2, "liquid_molar_density_mol_m3": 13000.0,
            "liquid_density_kg_m3": 780.0,
            "equilibrium": {"model": "antoine", "antoine": {c: dict(zip("ABC", ANT[c])) for c in comps}},
            "energy_balance": {"heat_duty_W": Q, "reference_temperature_K": T_REF,
                               "properties": {c: dict(zip(("cp_liquid_J_molK", "cp_vapour_J_molK",
                                                           "heat_of_vaporization_J_mol"), TH[c])) for c in comps}},
            "feed": {"molar_flow_mol_s": F, "mole_fractions": zz, "temperature_K": TF, "phase": "liquid"},
            "vapour_outlet": vap or {"type": "valve", "coefficient_mol_s_Pa05": 0.0094, "opening": 0.5,
                                     "characteristic": "linear", "downstream_pressure_Pa": 1e5},
            "liquid_outlet": liq or {"type": "valve", "Kv_m3_h": 7.45, "opening": 0.6, "characteristic": "linear",
                                     "downstream_pressure_Pa": 1e5},
            "initial": {"liquid_level_m": h0, "mole_fractions": dict(zip(comps, z0 or z)), "temperature_K": T0}}


def _designed(Q, P=1.2e5, h=0.8):
    """Drum whose valves are sized so that the steady state sits at P and level h (vapour valve C·0.5·√(P − 1 bar) =
    β·F, liquid valve at 0.6 opening passes (1 − β)·F); β from the module's own P-H flash of the feed at P."""
    d = _drum(Q=Q)
    probe = {"pressure_Pa": P, "temperature_K": 345.0, "vapour_flow_mol_s": 0.0}
    probe.update({f"{k}.{c}": 0.0 for k in "xy" for c in COMP})
    b = FD.ph_flash_check(FD.Drum(d), probe, True)["vapour_fraction"]
    d["vapour_outlet"]["coefficient_mol_s_Pa05"] = b * 10.0 / (0.5 * math.sqrt(P - 1e5))
    d["liquid_outlet"]["Kv_m3_h"] = (1 - b) * 10.0 / 13000.0 * 3600.0 / math.sqrt((P - 1e5 + 780.0 * 9.80665 * h)
                                                                                  / 1e5 / 0.78) / 0.6
    return d


@pytest.mark.parametrize("Q", [0.0, 1.5e5])
def test_flash_steady_state_equals_thermo_ph_flash(Q):
    """The steady state of the dynamic energy drum (adiabatic and heated) is the P-H flash of the feed at the drum
    pressure computed by thermo [THERMO] with H = H_feed + Q/F. Measured: |ΔT| ≤ 1.7e-13 K, |Δβ|, |Δx|, |Δy| ≤ 5e-14."""
    r = ok(FD.simulate_flash_drum(_designed(Q), 50.0, start_from_steady_state=True, n_points=11))
    assert r["at_steady_state"] and r["initial_steady_state"]["converged"]
    f = r["final_state"]
    fl = _thermo_flasher(13000.0)
    feed = fl.flash(T=360.0, P=5e5, zs=[0.6, 0.4])
    assert feed.phase == "L"
    st = fl.flash(P=f["pressure_Pa"], H=feed.H() + Q / 10.0, zs=[0.6, 0.4])
    assert f["temperature_K"] == pytest.approx(st.T, abs=1e-10)
    assert f["vapour_flow_mol_s"] / 10.0 == pytest.approx(st.VF, abs=1e-11)
    assert [f["x.acetone"], f["y.acetone"]] == pytest.approx([st.liquids[0].zs[0], st.gas.zs[0]], abs=1e-11)
    assert r["ph_flash_check"]["max_deviation"] < 1e-11 and r["rachford_rice_check"]["max_deviation"] < 1e-11
    if Q > 0:                                                    # the heater vaporises more and the drum is hotter
        assert f["vapour_flow_mol_s"] > 2.0                     # ≈ 0.66 mol/s without the heater
        assert f["pressure_Pa"] == pytest.approx(1.2e5, rel=1e-9) and f["liquid_level_m"] == pytest.approx(0.8, rel=1e-9)


def test_flash_transient_energy_closure_and_new_steady_state():
    """Heater step and feed-temperature step from the adiabatic steady state: ∫(F h_F − V h_V − L h_L + Q)dt = ΔU,
    T rises right after the duty step, and the end state is the new P-H flash. Measured: closure 7e-17, algebraic
    residual 3e-17, refinement 1e-6 of the span, P-H deviation 4e-10 (slow level/composition tail: 20000 s)."""
    ev = [{"time_s": 50.0, "target": "heat_duty_W", "value": 3.0e3},
          {"time_s": 400.0, "target": "feed.temperature_K", "value": 358.0}]
    r = ok(FD.simulate_flash_drum(_designed(0.0), 20000.0, events=ev, start_from_steady_state=True, n_points=401))
    e = r["evidence"]
    assert e["balances"]["energy_J"]["closure_rel"] < 1e-12
    assert e["algebraic_energy_residual_rel"] < 1e-12
    assert e["worst_balance_closure_rel"] < 1e-9
    assert e["balances"]["energy_J"]["heat_added"] == pytest.approx(3e3 * 19950.0, rel=1e-12)
    t, T = np.array(r["time_s"]), np.array(r["series"]["temperature_K"])
    i50 = np.searchsorted(t, 50.0)
    assert np.all(np.abs(T[:i50] - T[0]) < 1e-7)                 # steady before the step
    assert T[i50 + 2] > T[0]
    assert r["at_steady_state"] and r["ph_flash_check"]["max_deviation"] < 1e-8
    assert e["tolerance_refinement"]["max_difference_rel_to_span"] < 1e-4


def test_flash_pure_component_adiabatic_limit_is_closed_form():
    """Acetone with a 1e-7 trace of ethanol, liquid feed at 360 K, vapour draw fixed at β·F: at steady state the energy
    balance gives T − T_ref = [cp_L(T_F − T_ref) − β ΔH]/((1 − β) cp_L + β cp_V) and P = P_sat(T) (Antoine) in closed
    form. Measured: |ΔT| 3.3e-7 K, |ΔP|/P 6.3e-8 (both O(trace)); level 1.5e-6 m from the valve sizing."""
    beta, F, TF = 0.05, 10.0, 360.0
    cl, cv, dh = TH["acetone"]
    T = T_REF + (cl * (TF - T_REF) - beta * dh) / ((1 - beta) * cl + beta * cv)
    P = psat("acetone", T)
    rho = 780.0
    kv = (1 - beta) * F / 13000.0 * 3600.0 / math.sqrt((P + rho * 9.80665 * 0.8 - 1e5) / 1e5 / (rho / 1000.0)) / 0.6
    d = _drum(TF=TF, z=(1 - 1e-7, 1e-7), F=F, T0=T,
              vap={"type": "fixed", "molar_flow_mol_s": beta * F},
              liq={"type": "valve", "Kv_m3_h": kv, "opening": 0.6, "characteristic": "linear",
                   "downstream_pressure_Pa": 1e5})
    r = ok(FD.simulate_flash_drum(d, 100.0, start_from_steady_state=True, n_points=5))
    f = r["final_state"]
    assert f["temperature_K"] == pytest.approx(T, abs=2e-6)
    assert f["pressure_Pa"] == pytest.approx(P, rel=5e-7)
    assert f["liquid_level_m"] == pytest.approx(0.8, abs=1e-5)


def test_flash_closed_drum_heating_matches_thermo_volume_and_energy():
    """Closed rigid drum (no feed, outlets shut) heated at Q: U(t) = U0 + Q·t exactly, and the (T, P) the model finds
    from (M, U) must give, in thermo's independent TP flash [THERMO], the drum volume and the same energy change.
    Asserted 1e-9 (measured 2e-11 and 2e-12 in a 5e4 W variant); T rises 330 → 363.8 K, P to 2.37 bar."""
    Q, rhom = 1.0e4, 13000.0
    d = _drum(Q=Q, F=0.0, vap={"type": "fixed", "molar_flow_mol_s": 0.0}, liq={"type": "fixed", "molar_flow_mol_s": 0.0},
              T0=330.0, h0=0.5)
    r = ok(FD.simulate_flash_drum(d, 600.0, n_points=7))
    dr = FD.Drum(d)
    N, z = float(dr.M0.sum()), dr.M0 / dr.M0.sum()
    fl = _thermo_flasher(rhom)
    U0 = None
    for k, t in enumerate(r["time_s"]):
        T, P = r["series"]["temperature_K"][k], r["series"]["pressure_Pa"][k]
        st = fl.flash(T=T, P=P, zs=list(z))
        assert st.V() * N == pytest.approx(0.6, rel=1e-9)
        assert st.VF == pytest.approx(r["series"]["holdup_vapour_fraction"][k], abs=1e-9)
        if U0 is None:
            U0 = st.U() * N
        assert st.U() * N - U0 == pytest.approx(Q * t, rel=1e-9, abs=1e-3)
    T = np.array(r["series"]["temperature_K"])
    assert np.all(np.diff(T) > 0) and np.all(np.diff(r["series"]["pressure_Pa"]) > 0)
    assert r["evidence"]["balances"]["energy_J"]["accumulated"] == pytest.approx(Q * 600.0, rel=1e-12)


def test_flash_isothermal_antoine_model_equals_relative_volatility_model():
    """Isothermal mode with the new 'antoine' equilibrium reproduces the Kantor drum written with α and P_ref."""
    base = {"components": COMP, "temperature_K": 338.15, "volume_m3": 0.6, "area_m2": 0.2,
            "liquid_molar_density_mol_m3": 13000.0, "liquid_density_kg_m3": 780.0,
            "feed": {"molar_flow_mol_s": 10.0, "mole_fractions": {"acetone": 0.6, "ethanol": 0.4}},
            "vapour_outlet": {"type": "fixed", "molar_flow_mol_s": 2.317},
            "liquid_outlet": {"type": "valve", "Kv_m3_h": 10.0, "opening": 0.6, "characteristic": "linear",
                              "downstream_pressure_Pa": 1e5},
            "initial": {"liquid_level_m": 0.8, "mole_fractions": {"acetone": 0.6, "ethanol": 0.4}}}
    a = dict(base, equilibrium={"model": "antoine", "antoine": {c: dict(zip("ABC", ANT[c])) for c in COMP}})
    pa, pb = psat("acetone", 338.15), psat("ethanol", 338.15)
    b = dict(base, equilibrium={"model": "relative_volatility", "alpha": {"acetone": pa / pb, "ethanol": 1.0},
                                "reference_vapour_pressure_Pa": pb})
    ra, rb = ok(FD.simulate_flash_drum(a, 2000.0, n_points=21)), ok(FD.simulate_flash_drum(b, 2000.0, n_points=21))
    for k in ("pressure_Pa", "liquid_level_m", "x.acetone"):
        assert ra["series"][k] == pytest.approx(rb["series"][k], rel=1e-9)
    assert "temperature_K" not in ra["final_state"]                       # isothermal output unchanged
    assert any("modo isotérmico" in s for s in ra["not_included"])


def test_flash_heated_drum_that_runs_dry_is_reported():
    """With a large heater and the adiabatic valve sizing the liquid valve drains the drum at the higher pressure: the
    run stops at the edge of the two-phase region and says so (never continues on a fictitious state)."""
    with pytest.raises(ValueError, match="región de dos fases"):
        FD.simulate_flash_drum(_drum(Q=1.5e5), 8000.0, n_points=5, refine=False)


def test_flash_energy_refusals():
    with pytest.raises(ValueError, match="heat_duty_W"):
        d = _drum()
        del d["energy_balance"]["heat_duty_W"]
        FD.simulate_flash_drum(d, 10.0)
    with pytest.raises(ValueError, match="feed.temperature_K|temperature_K"):
        d = _drum()
        del d["feed"]["temperature_K"]
        FD.simulate_flash_drum(d, 10.0)
    with pytest.raises(ValueError, match="phase"):
        d = _drum()
        del d["feed"]["phase"]
        FD.simulate_flash_drum(d, 10.0)
    with pytest.raises(ValueError, match="es un estado"):
        FD.simulate_flash_drum(dict(_drum(), temperature_K=340.0), 10.0)
    with pytest.raises(ValueError, match="initial"):
        d = _drum()
        del d["initial"]["temperature_K"]
        FD.simulate_flash_drum(d, 10.0)
    with pytest.raises(ValueError, match="antoine"):
        FD.simulate_flash_drum(dict(_drum(), equilibrium={"model": "k_values", "K": {"acetone": 2.0, "ethanol": 0.5}}),
                               10.0)
    with pytest.raises(ValueError, match="heat_of_vaporization_J_mol"):
        d = _drum()
        del d["energy_balance"]["properties"]["ethanol"]["heat_of_vaporization_J_mol"]
        FD.simulate_flash_drum(d, 10.0)
    fs = {"components": COMP, "units": [dict(_drum(), id="FL", type="flash_drum")]}
    with pytest.raises(ValueError, match="solo está disponible en dynamic_flash_drum"):
        FS.simulate(fs, 10.0)


# ------------------------------------------------------------------------------------------------------ column
LAM = 30000.0


def _cmo_energy(fs):
    """Skogestad column A rewritten in energy mode at the constant-molar-overflow limit: equal latent heats, zero heat
    capacities, and Antoine constants with the same B and C so that P_l/P_h = 10^(A_l − A_h) = α = 1.5 exactly."""
    fs = copy.deepcopy(fs)
    u = fs["units"][0]
    vb = u.pop("boilup_mol_s")
    u.pop("relative_volatility")
    u["reboiler_duty_W"] = vb * LAM

    def comp(A):
        return {"antoine": {"A": A, "B": 1500.0, "C": -50.0}, "cp_liquid_J_molK": 0.0, "cp_vapour_J_molK": 0.0,
                "heat_of_vaporization_J_mol": LAM}

    u["energy_balance"] = {"pressure_Pa": 101325.0, "reference_temperature_K": 350.0, "light": comp(10.0 + math.log10(1.5)),
                           "heavy": comp(10.0)}
    return fs


def test_column_energy_cmo_limit_reproduces_skogestad_a():
    """CMO limit of the energy column = existing CMO column [SKA]: steady profile (41 values, measured 1.1e-12) and the
    open-loop +1 % feed transient (measured: reboiler holdup 9.5e-7 mol of ≈ 590, x_B 4.3e-11, x_D 1.8e-11; both runs
    are integrated separately, so this is integration error at rtol 1e-8)."""
    r = ok(TDF.dynamic_flowsheet_steady_state(_cmo_energy(_cola()), outputs=STAGES))
    assert [r["outputs"][n] for n in STAGES] == pytest.approx(COLA_X, abs=1e-10)
    ev = [{"time_s": 0.0, "target": "C.feed_mol_s", "value": 1.01 * KMIN}]
    outs = ["C.reboiler_holdup_mol", "C.bottoms_x", "C.distillate_x"]
    a = ok(TDF.dynamic_flowsheet_run(_cola(None), 600.0, events=ev, outputs=outs, n_points=61))
    b = ok(TDF.dynamic_flowsheet_run(_cmo_energy(_cola(None)), 600.0, events=ev,
                                     outputs=outs + ["C.reboiler_vapour_mol_s", "C.top_vapour_mol_s"], n_points=61))
    assert b["series"]["C.reboiler_holdup_mol"] == pytest.approx(a["series"]["C.reboiler_holdup_mol"], abs=1e-5)
    for o in ("C.bottoms_x", "C.distillate_x"):
        assert b["series"][o] == pytest.approx(a["series"][o], abs=1e-9)
    vb = 3.20629295971439 * KMIN
    assert b["series"]["C.reboiler_vapour_mol_s"] == pytest.approx([vb] * 61, rel=1e-12)
    assert b["series"]["C.top_vapour_mol_s"] == pytest.approx([vb] * 61, rel=1e-12)


# acetone (light) / ethanol (heavy) column, illustrative design: 12 stages, feed on 6, 1 atm
NT, NF, F, ZF, LT, QR, PCOL = 12, 6, 10.0, 0.5, 12.0, 6.0e5, 101325.0


def _ecol(feed=None, x0=None, qr=QR):
    def comp(c):
        return {"antoine": dict(zip("ABC", ANT[c])), "cp_liquid_J_molK": TH[c][0], "cp_vapour_J_molK": TH[c][1],
                "heat_of_vaporization_J_mol": TH[c][2]}
    u = {"id": "C", "type": "distillation_column", "stages": NT, "feed_stage": NF,
         "hydraulics": {"tau_L_s": 5.0, "lambda": 0.0, "nominal_holdup_mol": 200.0, "nominal_liquid_flow_top_mol_s": LT,
                        "nominal_liquid_flow_bottom_mol_s": LT + F},
         "feed": feed or {"molar_flow_mol_s": F, "mole_fraction_light": ZF, "temperature_K": 320.0},
         "reflux_mol_s": LT, "reboiler_duty_W": qr, "distillate_mol_s": 5.0, "bottoms_mol_s": 5.0,
         "initial": {"x": x0 or list(np.linspace(0.1, 0.9, NT)), "holdup_mol": 200.0},
         "energy_balance": {"pressure_Pa": PCOL, "reference_temperature_K": T_REF, "light": comp("acetone"),
                            "heavy": comp("ethanol")}}
    ctl = [{"id": "LD", "measurement": "C.condenser_holdup_mol", "setpoint": 200.0, "Kc": -0.2, "tau_I_s": 50.0,
            "bias": 5.0, "manipulated": "C.distillate_mol_s"},
           {"id": "LB", "measurement": "C.reboiler_holdup_mol", "setpoint": 200.0, "Kc": -0.2, "tau_I_s": 50.0,
            "bias": 5.0, "manipulated": "C.bottoms_mol_s"}]
    return {"units": [u], "controllers": ctl}


def _hl(x, T):
    return (x * TH["acetone"][0] + (1 - x) * TH["ethanol"][0]) * (T - T_REF)


def _hv(y, T):
    (la, ca), (le, ce) = (TH["acetone"][2], TH["acetone"][1]), (TH["ethanol"][2], TH["ethanol"][1])
    return y * (la + ca * (T - T_REF)) + (1 - y) * (le + ce * (T - T_REF))


def _tbub(x):
    return brentq(lambda T: x * psat("acetone", T) + (1 - x) * psat("ethanol", T) - PCOL, 250.0, 450.0, xtol=1e-13)


def _feed_h(q):
    """Saturated feed of liquid fraction q at PCOL: independent flash at fixed vapour fraction."""
    b, z = 1 - q, np.array([ZF, 1 - ZF])

    def k(T):
        return np.array([psat("acetone", T), psat("ethanol", T)]) / PCOL

    T = brentq(lambda T: float(np.sum(z * (k(T) - 1) / (1 + b * (k(T) - 1)))), 300.0, 400.0, xtol=1e-13)
    x = z / (1 + b * (k(T) - 1))
    return (1 - b) * _hl(x[0], T) + b * _hv(k(T)[0] * x[0], T)


def _mesh(hF):
    """Own steady MESH solve (independent of packages.dynamics.column): unknowns x (NT) and V (NT − 1); L from the
    total balances from the top, B = L2 − V1, D = V_top − L_T; component and energy balances on every stage."""
    def unpack(u):
        x, V = u[:NT], u[NT:]
        T = np.array([_tbub(a) for a in x])
        y = x * np.array([psat("acetone", t) for t in T]) / PCOL
        L = np.zeros(NT)
        L[NT - 1] = LT
        for j in range(NT - 2, 0, -1):
            L[j] = L[j + 1] + V[j - 1] - V[j] + (F if j == NF - 1 else 0.0)
        return x, V, T, y, L, L[1] - V[0], V[NT - 2] - LT

    def res(u):
        x, V, T, y, L, B, D = unpack(u)
        h, H = _hl(x, T), _hv(y, T)
        r = [L[1] * x[1] - V[0] * y[0] - B * x[0]]
        r += [L[j + 1] * x[j + 1] + V[j - 1] * y[j - 1] + (F * ZF if j == NF - 1 else 0) - L[j] * x[j] - V[j] * y[j]
              for j in range(1, NT - 1)]
        r += [V[NT - 2] * y[NT - 2] - (LT + D) * x[NT - 1], (L[1] * h[1] - V[0] * H[0] - B * h[0] + QR) / 1e4]
        r += [(L[j + 1] * h[j + 1] + V[j - 1] * H[j - 1] + (F * hF if j == NF - 1 else 0) - L[j] * h[j] - V[j] * H[j]) / 1e4
              for j in range(1, NT - 1)]
        return np.array(r)

    s = root(res, np.concatenate([np.linspace(0.05, 0.8, NT), np.full(NT - 1, QR / 35000.0)]), method="hybr",
             options={"xtol": 1e-14})
    assert np.max(np.abs(res(s.x))) < 1e-10
    x, V, T, y, L, B, D = unpack(s.x)
    return x, V, T, D, B, V[NT - 2] * _hv(y[NT - 2], T[NT - 2]) - (LT + D) * _hl(x[NT - 1], T[NT - 1])


OUTS = ([f"C.x.{i}" for i in range(1, NT + 1)] + [f"C.temperature_K.{i}" for i in range(1, NT + 1)]
        + [f"C.vapour_flow_mol_s.{i}" for i in range(1, NT)] + ["C.condenser_duty_W", "C.distillate_mol_s",
                                                               "C.bottoms_mol_s"])


@pytest.mark.parametrize("feed", ["subcooled", "two_phase"])
def test_column_energy_steady_state_equals_own_mesh(feed):
    """Steady state of the dynamic energy column (Newton on the dynamic equations, level PI loops) = own steady MESH
    with energy balances. Measured: x 2.3e-15, V 1.1e-15 relative, T 1.1e-13 K, D and Q_C 1.3e-15 relative; stage
    temperatures vs thermo bubble points 3e-8 K (thermo's own tolerance). V_max/V_min = 1.15 and 1.38: not CMO."""
    fd = {"molar_flow_mol_s": F, "mole_fraction_light": ZF}
    fd.update({"temperature_K": 320.0} if feed == "subcooled" else {"liquid_fraction": 0.6})
    hF = _hl(ZF, 320.0) if feed == "subcooled" else _feed_h(0.6)
    r = ok(TDF.dynamic_flowsheet_steady_state(_ecol(feed=fd), outputs=OUTS))
    assert r["converged"] and r["stable"]
    o = r["outputs"]
    x, V, T, D, B, QC = _mesh(hF)
    assert [o[f"C.x.{i}"] for i in range(1, NT + 1)] == pytest.approx(list(x), abs=1e-10)
    assert [o[f"C.vapour_flow_mol_s.{i}"] for i in range(1, NT)] == pytest.approx(list(V), rel=1e-10)
    assert [o[f"C.temperature_K.{i}"] for i in range(1, NT + 1)] == pytest.approx(list(T), abs=1e-8)
    assert [o["C.distillate_mol_s"], o["C.bottoms_mol_s"], o["C.condenser_duty_W"]] == pytest.approx([D, B, QC], rel=1e-10)
    assert max(V) / min(V) > 1.05                                # energy balances move V by > 5 % along the column
    fl = _thermo_flasher(13000.0)                               # stage temperatures = thermo bubble points [THERMO]
    for i in (1, NF, NT):
        xi = o[f"C.x.{i}"]
        assert o[f"C.temperature_K.{i}"] == pytest.approx(fl.flash(P=PCOL, VF=0.0, zs=[xi, 1 - xi]).T, abs=2e-7)


def test_column_energy_transient_closure_and_duty_response():
    """From the steady state: +5 % reboiler duty at 100 s, feed 10 K warmer at 600 s. The liquid holdup enthalpy is not
    a state, so ∫(F h_F + Q_R − Q_C − D h_D − B h_B) dt = ΔΣ M h is a genuine check of the analytical index reduction.
    Measured: energy closure 4.8e-15, component closures ≤ 6e-16, refinement 8e-10 of the span. The boilup jumps at
    once by ΔQ_R/[H − h − h'(y − x)] of the reboiler, recomputed here independently (h' by central differences
    along the bubble curve): measured agreement 9e-12 relative."""
    ev = [{"time_s": 100.0, "target": "C.reboiler_duty_W", "value": 1.05 * QR},
          {"time_s": 600.0, "target": "C.feed_temperature_K", "value": 330.0}]
    outs = ["C.reboiler_vapour_mol_s", "C.top_vapour_mol_s", "C.condenser_duty_W", "C.distillate_x", "C.bottoms_x",
            "C.top_temperature_K", "C.bottom_temperature_K"]
    r = ok(TDF.dynamic_flowsheet_run(_ecol(), 2000.0, events=ev, outputs=outs, start_from_steady_state=True, n_points=201))
    b = r["evidence"]["balances"]
    assert b["column_energy_J"]["closure_rel"] < 1e-12
    assert b["column_total_mol"]["closure_rel"] < 1e-13 and b["column_light_mol"]["closure_rel"] < 1e-13
    t, vb = np.array(r["time_s"]), np.array(r["series"]["C.reboiler_vapour_mol_s"])
    i = np.searchsorted(t, 100.0)
    assert np.all(np.abs(vb[:i] - vb[0]) < 1e-8 * vb[0])
    ss = TDF.dynamic_flowsheet_steady_state(_ecol(), outputs=["C.x.1"])["outputs"]["C.x.1"]
    t1 = _tbub(ss)
    y1 = ss * psat("acetone", t1) / PCOL
    dhdx = (_hl(ss + 1e-6, _tbub(ss + 1e-6)) - _hl(ss - 1e-6, _tbub(ss - 1e-6))) / 2e-6   # along the bubble curve
    d0 = _hv(y1, t1) - _hl(ss, t1) - dhdx * (y1 - ss)
    assert vb[i] - vb[0] == pytest.approx(0.05 * QR / d0, rel=1e-9)         # holdups are continuous: ΔV_B = ΔQ_R/d0
    assert r["series"]["C.distillate_x"][-1] != pytest.approx(r["series"]["C.distillate_x"][0], abs=1e-6)


def test_column_energy_refusals():
    def bad(mut, match):
        fs = _ecol()
        mut(fs["units"][0])
        with pytest.raises(ValueError, match=match):
            FS.simulate(fs, 10.0)

    bad(lambda u: u.update(relative_volatility=1.5), "quite 'relative_volatility'")
    bad(lambda u: u.update(boilup_mol_s=10.0), "reboiler_duty_W")
    bad(lambda u: u.pop("reboiler_duty_W"), "reboiler_duty_W")
    bad(lambda u: u["feed"].update(liquid_fraction=1.0), "exactamente uno")
    bad(lambda u: u["feed"].pop("temperature_K"), "exactamente uno")
    bad(lambda u: u["energy_balance"].pop("pressure_Pa"), "pressure_Pa")
    bad(lambda u: u["energy_balance"]["heavy"].pop("cp_liquid_J_molK"), "cp_liquid_J_molK")
    bad(lambda u: u["energy_balance"].update(light=u["energy_balance"]["heavy"], heavy=u["energy_balance"]["light"]),
        "más volátil")
    fs = _cola(None)
    fs["units"][0].pop("relative_volatility")
    with pytest.raises(ValueError, match="relative_volatility"):
        FS.simulate(fs, 10.0)
