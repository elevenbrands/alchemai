"""Non-isothermal tubular reactor (bloque M7): 1-D PFR against a published model, evidence logic, and the FEniCSx limits.

Published case: Rawlings & Ekerdt, Chemical Reactor Analysis and Design Fundamentals, fig. 6.35 (o-xylene oxidation
in a cooled fixed-bed tube, parametric sensitivity). Their code (xyleneT.m, GPL) is ported verbatim below as
`rawlings_xylene`; its numbers are not printed in the book page (only a figure), so the reference is their model.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.transport import reactor_field as RF  # noqa: E402
from packages.transport import reactor_nonisothermal as N  # noqa: E402

# --- Rawlings & Ekerdt fig. 6.35, verbatim (units: kmol, kJ, kN/m², m, s) -------------------------------------------
R_TUBE, QRHO, PF, RG, E_K, TM, KM, TA, CP, DELH, U_RE, LEN, YX = 1.25e-2, 0.0026371, 1.01e2, 8.314, 13636., 625., 2.0822, \
    625., 0.992, -1.284e6, 0.373, 1.5, 0.019
MWF = 0.98 * (0.79 * 28 + 0.21 * 32) + 0.02 * 106.17


def rawlings_xylene(Tf: float) -> dict:
    Ac = math.pi * R_TUBE ** 2
    Nf = QRHO / MWF
    beta, Gamma = DELH * Ac / (QRHO * CP), 2 * math.pi * R_TUBE * U_RE / (QRHO * CP)

    def rhs(z, s):
        Nx, T = s
        Q = Nf / (PF / (RG * T))
        rate = KM * math.exp(-E_K * (1 / T - 1 / TM)) * Nx / Q
        return [-rate * Ac, -beta * rate + Gamma * (TA - T)]
    Nxf = YX * Nf
    sol = solve_ivp(rhs, (0, LEN), [Nxf, Tf], method="Radau", rtol=1e-10, atol=1e-14, dense_output=True)
    z = np.linspace(0, LEN, 3001)
    y = sol.sol(z)
    i = int(y[1].argmax())
    return {"X": 1 - y[0, -1] / Nxf, "T_max_K": float(y[1, i]), "z_T_max_m": float(z[i])}


def xylene_inputs(Tf: float, lam: float, Dr: float) -> dict:
    """The same reactor in SI for the tool: ρ and C0 of the feed (ideal gas), A from km at Tm, plug flow, coolant."""
    P = PF * 1e3
    rho = P * MWF / 1e3 / (RG * Tf)
    return dict(volumetric_flow_m3_s=QRHO / rho, tube_diameter_m=2 * R_TUBE, n_tubes=1,
                inlet_concentration_mol_m3=YX * P / (RG * Tf), inlet_T_K=Tf, A_1_s=KM * math.exp(E_K / TM),
                E_J_mol=E_K * N.R_GAS, dH_J_mol=DELH, density_kg_m3=rho, cp_J_kgK=CP * 1e3,
                diffusivity_m2_s=Dr, thermal_conductivity_W_mK=lam, axial_diffusivity_m2_s=1e-5,
                axial_thermal_conductivity_W_mK=0.05, wall={"mode": "coolant", "U_W_m2K": U_RE * 1e3, "T_coolant_K": TA},
                velocity_profile="plug", fluid_model="ideal_gas", tube_length_m=LEN)


def pfr_for(inp: dict, U: float) -> dict:
    Ac = math.pi * inp["tube_diameter_m"] ** 2 / 4
    return N.pfr_1d(A_1_s=inp["A_1_s"], E_J_mol=inp["E_J_mol"], dH_J_mol=inp["dH_J_mol"],
                    C0=inp["inlet_concentration_mol_m3"], T0=inp["inlet_T_K"], rho_cp=inp["density_kg_m3"] * inp["cp_J_kgK"],
                    u_mean=inp["volumetric_flow_m3_s"] / Ac, tube_diameter_m=inp["tube_diameter_m"],
                    length_m=inp["tube_length_m"], U_1D=U, T_sink=inp["wall"]["T_coolant_K"], ideal_gas=True)


@pytest.mark.parametrize("Tf", [615.0, 620.0, 625.0, 630.0])
def test_pfr_1d_reproduces_rawlings_ekerdt_model(Tf):
    ref = rawlings_xylene(Tf)
    got = pfr_for(xylene_inputs(Tf, 1.0, 1.0), U_RE * 1e3)
    assert got["X_outlet"] == pytest.approx(ref["X"], abs=1e-4)
    assert got["T_max_K"] == pytest.approx(ref["T_max_K"], abs=0.2)


def test_pfr_1d_adiabatic_follows_the_adiabatic_line():
    T0, C0, rcp, dH = 330.0, 1000.0, 4.0e6, -2.0e5
    p = N.pfr_1d(A_1_s=1e9, E_J_mol=70e3, dH_J_mol=dH, C0=C0, T0=T0, rho_cp=rcp, u_mean=0.02, tube_diameter_m=0.025,
                 length_m=2.0, U_1D=0.0, T_sink=0.0)
    assert p["T_outlet_K"] - T0 == pytest.approx((-dH) * C0 / rcp * p["X_outlet"], rel=1e-8)


def test_equivalent_U_lumps_the_radial_resistance():
    U, Ts, _ = N.pfr_1d_equivalent_U({"mode": "coolant", "U_W_m2K": 100.0, "T_coolant_K": 600.0}, "plug", 0.02, 0.5)
    assert 1 / U == pytest.approx(1 / 100.0 + 0.01 / (4 * 0.5)) and Ts == 600.0
    U, Ts, _ = N.pfr_1d_equivalent_U({"mode": "fixed_T", "T_wall_K": 500.0}, "laminar", 0.02, 0.6)
    assert U == pytest.approx(3.66 * 0.6 / 0.02) and Ts == 500.0
    assert N.pfr_1d_equivalent_U({"mode": "adiabatic"}, "plug", 0.02, 0.5)[0] == 0.0


def test_inputs_are_required_and_validated():
    base = xylene_inputs(615.0, 0.5, 1e-3)
    for bad in ({"wall": {"mode": "coolant", "U_W_m2K": 100.0}}, {"wall": {"mode": "hot"}},
                {"velocity_profile": "turbulent"}, {"fluid_model": "steam"}, {"cp_J_kgK": None},
                {"thermal_conductivity_W_mK": 0.0}, {"dH_J_mol": None}):
        with pytest.raises(ValueError):
            N.nonisothermal_tubular_reactor(**(base | bad))


def _res(dX=0.0, dT=0.0, dH=0.0, closure=1e-10):
    m = lambda x, t, h: {"X_outlet": x, "T_outlet_cup_K": t, "hot_spot": {"T_max_K": h},
                         "balances": {"energy_closure_rel": closure, "species_closure_rel": 1e-12}}
    return {"coarse": m(0.6, 640.0, 690.0), "fine": m(0.6 + dX, 640.0 + dT, 690.0 + dH)}


def test_evidence_seal():
    assert N.evidence_from(_res(), "plug", None, 615.0)["verified"]
    assert not N.evidence_from(_res(closure=1e-3), "plug", None, 615.0)["energy_closed"]
    assert not N.evidence_from(_res(dX=5e-3), "plug", None, 615.0)["mesh_independent"]
    assert N.evidence_from(_res(dH=1.4), "plug", None, 615.0)["mesh_independent"]      # 2 % of a 75 K rise = 1.5 K
    assert not N.evidence_from(_res(dH=1.6), "plug", None, 615.0)["mesh_independent"]
    assert not N.evidence_from(_res(), "laminar", None, 615.0)["verified"]              # regime unknown
    assert N.evidence_from(_res(), "laminar", 1500.0, 615.0)["verified"]
    no_mesh = {"coarse": _res()["coarse"]}
    assert not N.evidence_from(no_mesh, "plug", None, 615.0)["verified"]


def test_divergence_locates_the_split():
    z = np.linspace(0, 1, 101)
    pfr = {"z_m": z.tolist(), "X": z.tolist(), "T_K": (600 + 10 * z).tolist()}
    field = {"z_m": z.tolist(), "X_cup": (z - 0.05 * (z > 0.5)).tolist(), "T_cup_K": (600 + 10 * z + 5 * np.exp(-((z - 0.3) / 0.05) ** 2)).tolist()}
    d = N.divergence(field, pfr)
    assert d["z_max_dT_m"] == pytest.approx(0.3) and d["max_abs_dT_cup_K"] == pytest.approx(5.0)
    assert d["z_first_dX_over_0.01_m"] == pytest.approx(0.51) and d["dX_outlet"] == pytest.approx(-0.05)


def test_tool_is_registered():
    from packages.dwsim_client import tools as T  # noqa: F401  (tools imports tools_transport)
    from packages.dwsim_client import tools_transport as TT
    from packages.agent import agent as AG
    assert TT.nonisothermal_tubular_reactor in TT.TRANSPORT_TOOLS
    assert any(getattr(t, "name", getattr(t, "__name__", "")) == "nonisothermal_tubular_reactor" for t in AG.AGENT_TOOLS)


# --- FEniCSx --------------------------------------------------------------------------------------------------------
needs_fenicsx = pytest.mark.skipif(not RF.fenicsx_available(), reason="FEniCSx environment not installed")


@needs_fenicsx
def test_dH_zero_equals_the_isothermal_field():
    """ΔH = 0 and an adiabatic wall: T stays at T0 and the field must be M1's tubular_reactor_field (Case 1)."""
    Q = 10 * 8.314462618 * 600 / 1e5
    iso = RF.tubular_reactor_field(volumetric_flow_m3_s=Q, tube_diameter_m=0.02, n_tubes=1228, volume_m3=4.0, A_1_s=1e8,
                                   E_J_mol=1e5, T_K=600.0, diffusivity_m2_s=1e-5, density_kg_m3=1.16, viscosity_Pa_s=1.5e-5,
                                   nz=150, nr=15)
    r = N.nonisothermal_tubular_reactor(volumetric_flow_m3_s=Q, tube_diameter_m=0.02, n_tubes=1228, volume_m3=4.0,
                                        inlet_concentration_mol_m3=20.05, inlet_T_K=600.0, A_1_s=1e8, E_J_mol=1e5, dH_J_mol=0.0,
                                        density_kg_m3=1.16, cp_J_kgK=2000.0, diffusivity_m2_s=1e-5,
                                        thermal_conductivity_W_mK=0.05, wall={"mode": "adiabatic"},
                                        velocity_profile="laminar", fluid_model="liquid", viscosity_Pa_s=1.5e-5,
                                        nz=150, nr=15)
    assert r["X"] == pytest.approx(iso["X_laminar_fine_mesh"], abs=5e-4)
    assert r["hot_spot"]["T_max_K"] == pytest.approx(600.0, abs=1e-6) and r["T_outlet_K"] == pytest.approx(600.0, abs=1e-6)
    assert r["evidence"]["verified"], r["evidence"]


@needs_fenicsx
def test_adiabatic_plug_and_strongly_mixed_laminar_reach_the_1d_adiabatic_pfr():
    T0, E, C0, rho, cp, tau, L = 330.0, 70e3, 1000.0, 1000.0, 4000.0, 100.0, 2.0
    u = L / tau
    common = dict(volumetric_flow_m3_s=u * math.pi * 0.025 ** 2 / 4, tube_diameter_m=0.025, n_tubes=1,
                  inlet_concentration_mol_m3=C0, inlet_T_K=T0, A_1_s=0.5 / tau * math.exp(E / (N.R_GAS * T0)), E_J_mol=E,
                  dH_J_mol=-50 * rho * cp / C0, density_kg_m3=rho, cp_J_kgK=cp, wall={"mode": "adiabatic"},
                  fluid_model="liquid", tube_length_m=L, axial_diffusivity_m2_s=1e-9, axial_thermal_conductivity_W_mK=0.6,
                  viscosity_Pa_s=1e-3)
    plug = N.nonisothermal_tubular_reactor(**common, diffusivity_m2_s=1e-9, thermal_conductivity_W_mK=0.6,
                                           velocity_profile="plug", nz=100, nr=4)
    assert plug["X"] == pytest.approx(plug["X_pfr_1d"], abs=1e-4)
    assert plug["T_outlet_K"] == pytest.approx(plug["T_outlet_pfr_1d_K"], abs=0.05)
    assert plug["evidence"]["verified"], plug["evidence"]
    b = plug["energy_balance_bundle_kW"]
    assert b["heat_generated"] == pytest.approx(b["enthalpy_rise"], rel=1e-6) and b["heat_to_wall"] == 0.0
    # laminar with radial mixing number α·τ/R² ≈ 100: flat profiles, same answer within 2e-3
    lam = N.nonisothermal_tubular_reactor(**common, diffusivity_m2_s=1.6e-4, thermal_conductivity_W_mK=640.0,
                                          velocity_profile="laminar", nz=100, nr=10)
    assert lam["dimensionless"]["radial_heat_number"] > 50
    assert lam["X"] == pytest.approx(lam["X_pfr_1d"], abs=2e-3)
    assert lam["T_outlet_K"] == pytest.approx(lam["T_outlet_pfr_1d_K"], abs=0.2)


@needs_fenicsx
@pytest.mark.parametrize("Tf", [615.0, 625.0])
def test_strong_radial_mixing_reproduces_rawlings_ekerdt(Tf):
    """Flat radial profiles (λ_r, D_r huge, R/(4λ_r) negligible): the 2-D field must be the published 1-D model, even at
    625 K where 0.1 % in the rate constant moves the hot spot ~11 K."""
    ref = rawlings_xylene(Tf)
    r = N.nonisothermal_tubular_reactor(**xylene_inputs(Tf, 1e6, 1e3), nz=400, nr=2)
    assert r["X"] == pytest.approx(ref["X"], abs=1e-3)
    assert r["hot_spot"]["T_max_K"] == pytest.approx(ref["T_max_K"], abs=0.1)
    assert r["hot_spot"]["z_m"] == pytest.approx(ref["z_T_max_m"], abs=0.01)
    assert r["evidence"]["verified"], r["evidence"]


@needs_fenicsx
def test_fixed_wall_closure_uses_the_consistent_flux():
    """Exothermic liquid in a laminar tube with the wall held at T0: heat leaves through the wall, the field's hot
    spot on the axis is above the 1-D PFR's (lumped Nu = 3.66) and the balance closes with the residual wall flux."""
    T0, E, C0, rho, cp, tau, L, D = 330.0, 70e3, 1000.0, 1000.0, 4000.0, 200.0, 2.0, 0.01
    r = N.nonisothermal_tubular_reactor(volumetric_flow_m3_s=L / tau * math.pi * D ** 2 / 4, tube_diameter_m=D, n_tubes=10,
                                        inlet_concentration_mol_m3=C0, inlet_T_K=T0,
                                        A_1_s=0.01 * math.exp(E / (N.R_GAS * T0)), E_J_mol=E, dH_J_mol=-60 * rho * cp / C0,
                                        density_kg_m3=rho, cp_J_kgK=cp, diffusivity_m2_s=1e-9,
                                        thermal_conductivity_W_mK=0.6, wall={"mode": "fixed_T", "T_wall_K": T0},
                                        velocity_profile="laminar", fluid_model="liquid", tube_length_m=L,
                                        viscosity_Pa_s=1e-3, nz=100, nr=16)
    ev = r["evidence"]
    assert ev["energy_closed"] and ev["energy_closure_rel"] < N.ENERGY_TOL, ev
    b = r["energy_balance_bundle_kW"]
    assert b["heat_to_wall"] > 0 and b["heat_generated"] > b["enthalpy_rise"] > 0
    assert r["hot_spot"]["r_m"] < 0.2 * D / 2                          # hot spot on (or next to) the axis
    prof = r["radial_T_profile_hot_spot"]["T_K"]
    assert prof[0] > prof[-1] == pytest.approx(T0, abs=0.5)           # centre hotter than the wall
    assert ev["mesh_independent"], ev


@pytest.mark.skipif(os.environ.get("ALCHEMAI_SLOW_FEM") != "1" or not RF.fenicsx_available(),
                    reason="slow (~70 s): set ALCHEMAI_SLOW_FEM=1")
def test_cooled_laminar_tube_hot_spot_above_the_pfr(tmp_path):
    """ILLUSTRATIVE exothermic liquid (MOCKS.md SUPUESTOS-REACTOR-NO-ISOTERMICO): 50 laminar tubes of 2 cm × 0.3 m,
    ΔT_ad 60 K, coolant at T0 through U = 300 W/m²K. The axis runs ~13 K above the 1-D PFR's hot spot (Nu 3.66 lumped).
    Measured 2026-11-13: field 398.42 K at z ≈ 1 cm, PFR 385.38 K, mesh 200×20 vs 400×40 ΔT_hot 0.04 K, 70 s."""
    T0, E, C0, rho, cp, tau, L, D = 330.0, 70e3, 1000.0, 1000.0, 4000.0, 30.0, 0.3, 0.02
    fig = tmp_path / "m7.png"
    r = N.nonisothermal_tubular_reactor(volumetric_flow_m3_s=L / tau * math.pi * D ** 2 / 4, tube_diameter_m=D, n_tubes=50,
                                        inlet_concentration_mol_m3=C0, inlet_T_K=T0,
                                        A_1_s=0.01 * math.exp(E / (N.R_GAS * T0)), E_J_mol=E, dH_J_mol=-60 * rho * cp / C0,
                                        density_kg_m3=rho, cp_J_kgK=cp, diffusivity_m2_s=1e-9, thermal_conductivity_W_mK=0.6,
                                        wall={"mode": "coolant", "U_W_m2K": 300.0, "T_coolant_K": T0},
                                        velocity_profile="laminar", fluid_model="liquid", tube_length_m=L,
                                        viscosity_Pa_s=1e-3, nz=200, nr=20, figure_path=str(fig))
    assert r["evidence"]["verified"], r["evidence"]
    assert r["hot_spot"]["T_max_K"] == pytest.approx(398.4, abs=0.5) and r["hot_spot_pfr_1d"]["T_max_K"] == pytest.approx(385.4, abs=0.2)
    assert r["hot_spot_excess_vs_pfr_K"] > 10.0
    prof = r["radial_T_profile_hot_spot"]["T_K"]
    assert prof[0] - prof[-1] > 30.0
    assert fig.is_file() and fig.stat().st_size > 20_000
