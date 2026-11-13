"""Non-isothermal tubular reactor as a coupled field (bloque M7 of docs/MAPA_MOTORES.md).

M1 (`reactor_field`) solved the isothermal concentration field of one laminar tube. Here temperature is a second
field: first-order Arrhenius kinetics, heat of reaction, radial and axial conduction, and a wall that is held at a
temperature, exchanges heat with a coolant through U, or is adiabatic. FEniCSx solves the coupled 2-D axisymmetric
problem in a SEPARATE PROCESS (`fenicsx/nonisothermal_reactor.py`, launched by `reactor_field.run_solver`); this module
solves the 1-D non-isothermal PFR (ideal plug flow, same kinetics) with scipy, and reports where the two diverge.

Every kinetic and thermal parameter comes from the caller; there are no defaults. Modelling assumptions are
returned in `assumptions` (and registered in MOCKS.md as SUPUESTOS-REACTOR-NO-ISOTERMICO).

Evidence → seal (`verified` only when all hold):
  * energy_closed: |H_out − H_in + Q_wall − Q_gen| ≤ ENERGY_TOL · max(|Q_gen|, |ΔH_sens|, |Q_wall|) on both meshes,
    and the species balance F_in − F_out = ∫ r dV within SPECIES_TOL · F_in (discrete conservation: Newton
    converged, boundary terms right)
  * mesh_independent: fine mesh (2× in z and r) vs coarse: |ΔX| < MESH_X_TOL, |ΔT_out| < MESH_T_TOL_K and
    |ΔT_hot| < max(MESH_T_TOL_K, MESH_HOT_REL · (T_hot − T_in))
  * laminar_regime (laminar profile only): Re < 2300, which needs viscosity_Pa_s
Limits verified once in tests/test_reactor_nonisothermal.py: ΔH = 0 reproduces `tubular_reactor_field`; adiabatic
plug flow (and laminar flow with strong radial mixing) reproduces the 1-D adiabatic PFR solved here with scipy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from packages.transport import reactor_field as RF

R_GAS = RF.R_GAS
SOLVER = Path(__file__).with_name("fenicsx") / "nonisothermal_reactor.py"
ENERGY_TOL = 1e-5
SPECIES_TOL = 1e-5
MESH_X_TOL = 2e-3
MESH_T_TOL_K = 0.5
MESH_HOT_REL = 0.02
DIVERGENCE_X = 0.01
DIVERGENCE_T_K = 2.0
NU_LAMINAR_CONST_T = 3.66
WALL_MODES = ("fixed_T", "coolant", "adiabatic")


def _check_wall(wall: dict) -> dict:
    if not isinstance(wall, dict) or wall.get("mode") not in WALL_MODES:
        raise ValueError("wall = {'mode': 'fixed_T', 'T_wall_K': …} | {'mode': 'coolant', 'U_W_m2K': …, 'T_coolant_K': …} "
                         "| {'mode': 'adiabatic'}")
    m = wall["mode"]
    if m == "fixed_T":
        if not wall.get("T_wall_K") or wall["T_wall_K"] <= 0:
            raise ValueError("fixed_T needs T_wall_K > 0")
        return {"mode": m, "T_wall_K": float(wall["T_wall_K"])}
    if m == "coolant":
        if wall.get("U_W_m2K") is None or wall.get("T_coolant_K") is None or wall["U_W_m2K"] < 0 or wall["T_coolant_K"] <= 0:
            raise ValueError("coolant needs U_W_m2K ≥ 0 (wall-to-coolant) and T_coolant_K > 0")
        return {"mode": m, "U_W_m2K": float(wall["U_W_m2K"]), "T_coolant_K": float(wall["T_coolant_K"])}
    return {"mode": m}


def pfr_1d_equivalent_U(wall: dict, profile: str, tube_diameter_m: float, lambda_r: float) -> tuple[float, float, str]:
    """(U_1D W/m²K, T_sink K, how) for the 1-D PFR: the radial resistance of the fluid lumped in series with the wall.
    Plug: 1/U = 1/U_wall + R/(4 λ_r) (flat-velocity parabolic T profile, Froment). Laminar: inner h = 3.66 λ/D (Graetz,
    fully developed, constant wall T)."""
    R = tube_diameter_m / 2.0
    h_in = 4.0 * lambda_r / R if profile == "plug" else NU_LAMINAR_CONST_T * lambda_r / tube_diameter_m
    how_in = "R/(4 λ_r)" if profile == "plug" else "D/(3.66 λ)"
    m = wall["mode"]
    if m == "adiabatic":
        return 0.0, 0.0, "pared adiabática: U = 0"
    if m == "fixed_T":
        return h_in, wall["T_wall_K"], f"pared a T fija: 1/U = {how_in}"
    U = 0.0 if wall["U_W_m2K"] == 0 else 1.0 / (1.0 / wall["U_W_m2K"] + 1.0 / h_in)
    return U, wall["T_coolant_K"], f"1/U = 1/U_pared + {how_in}"


def pfr_1d(*, A_1_s: float, E_J_mol: float, dH_J_mol: float, C0: float, T0: float, rho_cp: float, u_mean: float,
           tube_diameter_m: float, length_m: float, U_1D: float, T_sink: float, ideal_gas: bool = False,
           n: int = 401) -> dict:
    """Ideal non-isothermal PFR (flat profiles, no dispersion): scipy Radau. State c = molar flow of A / feed
    volumetric flow; constant density → that is the concentration; ideal gas (ρu = G) → concentration = c·T0/T."""
    def rhs(z, s):
        c, T = s
        conc = max(c, 0.0) * (T0 / T if ideal_gas else 1.0)
        r = A_1_s * math.exp(-E_J_mol / (R_GAS * T)) * conc
        return [-r / u_mean, ((-dH_J_mol) * r - 4.0 * U_1D / tube_diameter_m * (T - T_sink)) / (rho_cp * u_mean)]
    sol = solve_ivp(rhs, (0.0, length_m), [C0, T0], method="Radau", dense_output=True, rtol=1e-10,
                    atol=[1e-12 * max(C0, 1.0), 1e-9])
    if not sol.success:
        raise RuntimeError(f"PFR 1-D (scipy) falló: {sol.message}")
    z = np.linspace(0.0, length_m, n)
    c, T = sol.sol(z)
    X = 1 - c / C0
    zf = np.linspace(0.0, length_m, 20 * (n - 1) + 1)        # the hot spot of a runaway is sharp: locate it finely
    Tf = sol.sol(zf)[1]
    i = int(np.argmax(Tf))
    return {"z_m": z.tolist(), "X": X.tolist(), "T_K": T.tolist(), "X_outlet": float(X[-1]), "T_outlet_K": float(T[-1]),
            "T_max_K": float(Tf[i]), "z_T_max_m": float(zf[i])}


def evidence_from(res: dict, profile: str, Re: float | None, T_in: float) -> dict:
    meshes = [m for m in ("coarse", "fine") if m in res]
    worst_e = max(abs(res[m]["balances"]["energy_closure_rel"]) for m in meshes)
    worst_s = max(abs(res[m]["balances"]["species_closure_rel"]) for m in meshes)
    ev = {"energy_closure_rel": worst_e, "species_closure_rel": worst_s,
          "energy_closed": worst_e <= ENERGY_TOL and worst_s <= SPECIES_TOL,
          "tolerances": {"energy_rel": ENERGY_TOL, "species_rel": SPECIES_TOL, "mesh_X": MESH_X_TOL,
                         "mesh_T_K": MESH_T_TOL_K, "mesh_hot_rel": MESH_HOT_REL}}
    reasons = []
    if not ev["energy_closed"]:
        reasons.append(f"el balance no cierra (energía {worst_e:.1e}, especie {worst_s:.1e} relativo)")
    if "fine" in res:
        c, f = res["coarse"], res["fine"]
        dX = abs(f["X_outlet"] - c["X_outlet"])
        dT = abs(f["T_outlet_cup_K"] - c["T_outlet_cup_K"])
        dH = abs(f["hot_spot"]["T_max_K"] - c["hot_spot"]["T_max_K"])
        hot_tol = max(MESH_T_TOL_K, MESH_HOT_REL * abs(f["hot_spot"]["T_max_K"] - T_in))
        ok = dX < MESH_X_TOL and dT < MESH_T_TOL_K and dH < hot_tol
        ev.update(mesh_independent=ok, mesh_delta={"X": dX, "T_outlet_K": dT, "T_hot_spot_K": dH, "hot_tol_K": hot_tol})
        if not ok:
            reasons.append(f"el resultado cambia con la malla (ΔX {dX:.1e}, ΔT_salida {dT:.2f} K, ΔT_caliente {dH:.2f} K); "
                           "sube nz/nr")
    else:
        ev["mesh_independent"] = None
        reasons.append("sin estudio de malla (mesh_study=False): resultado sin verificar")
    if profile == "laminar":
        ev["laminar_regime"] = (Re < RF.RE_LAMINAR) if Re is not None else None
        ev["Re"] = Re
        if Re is None:
            reasons.append("régimen no comprobado: da viscosity_Pa_s para confirmar Re < 2300 con el perfil de Poiseuille")
        elif Re >= RF.RE_LAMINAR:
            reasons.append(f"Re = {Re:.0f} ≥ 2300: el perfil de Poiseuille no aplica")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    return ev


def divergence(field_axial: dict, pfr: dict) -> dict:
    """Where the 2-D field (cup-mixing) and the 1-D PFR part ways, along z."""
    z = np.asarray(field_axial["z_m"])
    Xf, Tf = np.asarray(field_axial["X_cup"]), np.asarray(field_axial["T_cup_K"])
    Xp = np.interp(z, pfr["z_m"], pfr["X"])
    Tp = np.interp(z, pfr["z_m"], pfr["T_K"])
    dX, dT = Xf - Xp, Tf - Tp
    iT, iX = int(np.argmax(np.abs(dT))), int(np.argmax(np.abs(dX)))
    first = lambda mask: float(z[np.argmax(mask)]) if mask.any() else None
    return {"max_abs_dT_cup_K": float(abs(dT[iT])), "z_max_dT_m": float(z[iT]), "dT_cup_at_max_K": float(dT[iT]),
            "max_abs_dX": float(abs(dX[iX])), "z_max_dX_m": float(z[iX]),
            f"z_first_dX_over_{DIVERGENCE_X}_m": first(np.abs(dX) > DIVERGENCE_X),
            f"z_first_dT_over_{DIVERGENCE_T_K:g}K_m": first(np.abs(dT) > DIVERGENCE_T_K),
            "dX_outlet": float(dX[-1]), "dT_cup_outlet_K": float(dT[-1])}


def save_figure(path: str, title: str, res: dict, pfr: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["coarse"]
    g, ax_, rh = c["grid"], c["axial"], c["radial_hot_spot"]
    fig = plt.figure(figsize=(10, 8), dpi=150)
    a1 = fig.add_subplot(3, 1, 1)
    Z, Rr = np.meshgrid(g["z_m"], np.asarray(g["r_m"]) * 1000, indexing="ij")
    pc = a1.pcolormesh(Z, Rr, np.asarray(g["T_K"]), shading="gouraud", cmap="inferno")
    hs = c["hot_spot"]
    a1.plot([hs["z_m"]], [hs["r_m"] * 1000], "c+", ms=12, mew=2)
    a1.set_xlabel("z (m)")
    a1.set_ylabel("r (mm)")
    a1.set_title(title, fontsize=10)
    fig.colorbar(pc, ax=a1, label="T (K)", pad=0.01)
    a2 = fig.add_subplot(3, 1, 2)
    a2.plot(ax_["z_m"], ax_["T_centerline_K"], color="#C0392B", lw=1.8, label="Campo 2-D · eje")
    a2.plot(ax_["z_m"], ax_["T_cup_K"], color="#1C6E93", lw=2.2, label="Campo 2-D · mezcla de copa")
    a2.plot(pfr["z_m"], pfr["T_K"], color="k", ls="--", lw=1.5, label="PFR 1-D ideal")
    a2.set_xlabel("z (m)")
    a2.set_ylabel("T (K)")
    a2.grid(alpha=0.25)
    a2.legend(fontsize=8)
    a3 = fig.add_subplot(3, 2, 5)
    a3.plot(np.asarray(rh["r_m"]) * 1000, rh["T_K"], color="#C0392B", lw=2)
    a3.set_xlabel("r (mm)")
    a3.set_ylabel(f"T (K) en z = {rh['z_m']:.3g} m")
    a3.grid(alpha=0.25)
    a4 = fig.add_subplot(3, 2, 6)
    a4.plot(ax_["z_m"], ax_["X_cup"], color="#1C6E93", lw=2, label="Campo 2-D")
    a4.plot(pfr["z_m"], pfr["X"], color="k", ls="--", lw=1.5, label="PFR 1-D")
    a4.set_xlabel("z (m)")
    a4.set_ylabel("X")
    a4.grid(alpha=0.25)
    a4.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nonisothermal_tubular_reactor(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int,
                                  inlet_concentration_mol_m3: float, inlet_T_K: float,
                                  A_1_s: float, E_J_mol: float, dH_J_mol: float,
                                  density_kg_m3: float, cp_J_kgK: float,
                                  diffusivity_m2_s: float, thermal_conductivity_W_mK: float,
                                  wall: dict, velocity_profile: str, fluid_model: str,
                                  tube_length_m: float | None = None, volume_m3: float | None = None,
                                  axial_diffusivity_m2_s: float | None = None,
                                  axial_thermal_conductivity_W_mK: float | None = None,
                                  viscosity_Pa_s: float | None = None,
                                  nz: int = 300, nr: int = 30, mesh_study: bool = True,
                                  figure_path: str | None = None, timeout_s: int = RF.DEFAULT_TIMEOUT_S) -> dict:
    """Coupled concentration-temperature field of one tube of a bundle (FEniCSx, separate process) vs the 1-D
    non-isothermal PFR (scipy). See the module docstring for the seal."""
    if velocity_profile not in ("laminar", "plug"):
        raise ValueError("velocity_profile must be 'laminar' (Poiseuille, empty tube) or 'plug' (packed bed, effective D_er, λ_er)")
    if fluid_model not in ("liquid", "ideal_gas"):
        raise ValueError("fluid_model must be 'liquid' (constant density) or 'ideal_gas' (constant P, ρu = G constant)")
    gas = fluid_model == "ideal_gas"
    wall = _check_wall(wall)
    pos = dict(inlet_concentration_mol_m3=inlet_concentration_mol_m3, inlet_T_K=inlet_T_K, A_1_s=A_1_s,
               density_kg_m3=density_kg_m3, cp_J_kgK=cp_J_kgK, diffusivity_m2_s=diffusivity_m2_s,
               thermal_conductivity_W_mK=thermal_conductivity_W_mK)
    bad = [k for k, v in pos.items() if v is None or not v > 0]
    if bad or E_J_mol is None or E_J_mol < 0 or dH_J_mol is None:
        raise ValueError(f"datos obligatorios y positivos: {bad or ''} (E_J_mol ≥ 0, dH_J_mol con signo: < 0 exotérmica)")
    geo = RF.bundle_geometry(volumetric_flow_m3_s, tube_diameter_m, n_tubes, tube_length_m, volume_m3)
    assumptions = [
        "Reacción única A → productos, primer orden en A, k = A·e^(−E/RT) por unidad de volumen de reactor.",
        ("Gas ideal a P constante con reactivo diluido: ρu = G constante, c = C0·(flujo molar relativo)·T0/T; ρ·D, cp y λ "
         "constantes; caudal, ρ y C0 son los de la alimentación." if gas else
         "Líquido: ρ, cp, D y λ constantes (sin dependencia con T ni composición); ū constante."),
        "Estado estacionario; entrada de Danckwerts (c y T de alimentación uniformes); salida con gradiente nulo.",
    ]
    if velocity_profile == "plug":
        assumptions.append("Perfil plano (modelo pseudo-homogéneo 2-D de lecho fijo): D y λ son los efectivos radiales "
                           "(D_er, λ_er) del usuario; resistencia de pared α_w incluida en U si se da modo coolant.")
    if wall["mode"] == "coolant":
        assumptions.append("Refrigerante a T constante a lo largo del tubo (ebullición o caudal grande); U = pared→refrigerante.")
    D_z = axial_diffusivity_m2_s
    if D_z is None:
        D_z = diffusivity_m2_s
        assumptions.append("Difusividad axial = radial (isótropa): no se dio axial_diffusivity_m2_s.")
    l_z = axial_thermal_conductivity_W_mK
    if l_z is None:
        l_z = thermal_conductivity_W_mK
        assumptions.append("Conductividad axial = radial (isótropa): no se dio axial_thermal_conductivity_W_mK.")
    U1, Tsink, how = pfr_1d_equivalent_U(wall, velocity_profile, tube_diameter_m, thermal_conductivity_W_mK)
    assumptions.append(f"PFR 1-D de comparación: perfiles planos, sin dispersión; U equivalente {U1:.4g} W/m²K ({how}).")
    Re = None
    if viscosity_Pa_s is not None:
        Re = density_kg_m3 * geo["u_mean_m_s"] * tube_diameter_m / viscosity_Pa_s
    payload = {"A_1_s": A_1_s, "E_J_mol": E_J_mol, "dH_J_mol": dH_J_mol, "C0_mol_m3": inlet_concentration_mol_m3,
               "T0_K": inlet_T_K, "rho_kg_m3": density_kg_m3, "cp_J_kgK": cp_J_kgK, "D_r": diffusivity_m2_s, "D_z": D_z,
               "lambda_r": thermal_conductivity_W_mK, "lambda_z": l_z, "tube_radius_m": tube_diameter_m / 2.0,
               "tube_length_m": geo["tube_length_m"], "u_mean_m_s": geo["u_mean_m_s"], "profile": velocity_profile,
               "ideal_gas": gas, "wall": wall, "nz": int(nz), "nr": int(nr), "mesh_study": bool(mesh_study)}
    res = RF.run_solver(payload, timeout_s=timeout_s, solver=SOLVER)
    rho_cp = density_kg_m3 * cp_J_kgK
    pfr = pfr_1d(A_1_s=A_1_s, E_J_mol=E_J_mol, dH_J_mol=dH_J_mol, C0=inlet_concentration_mol_m3, T0=inlet_T_K,
                 rho_cp=rho_cp, u_mean=geo["u_mean_m_s"], tube_diameter_m=tube_diameter_m, length_m=geo["tube_length_m"],
                 U_1D=U1, T_sink=Tsink, ideal_gas=gas)
    ev = evidence_from(res, velocity_profile, Re, inlet_T_K)
    c = res["coarse"]
    best = res.get("fine", c)
    bal = best["balances"]
    nt = n_tubes
    tau = geo["tau_s"]
    k_in = A_1_s * math.exp(-E_J_mol / (R_GAS * inlet_T_K))
    dT_ad = (-dH_J_mol) * inlet_concentration_mol_m3 / rho_cp
    R = tube_diameter_m / 2.0
    out = {
        "geometry": geo, "Re": Re, "velocity_profile": velocity_profile, "fluid_model": fluid_model, "wall": wall,
        "X": best["X_outlet"], "T_outlet_K": best["T_outlet_cup_K"],
        "hot_spot": best["hot_spot"],
        "X_pfr_1d": pfr["X_outlet"], "T_outlet_pfr_1d_K": pfr["T_outlet_K"],
        "hot_spot_pfr_1d": {"T_max_K": pfr["T_max_K"], "z_m": pfr["z_T_max_m"]},
        "hot_spot_excess_vs_pfr_K": best["hot_spot"]["T_max_K"] - pfr["T_max_K"],
        "divergence": divergence(c["axial"], pfr),
        "radial_T_profile_hot_spot": c["radial_hot_spot"], "radial_T_profile_outlet": c["radial_outlet"],
        "axial_profiles": c["axial"],
        "pfr_1d_profiles": {k: pfr[k][::4] for k in ("z_m", "X", "T_K")},
        "energy_balance_bundle_kW": {
            "heat_generated": nt * bal["Q_gen_W"] / 1e3, "enthalpy_rise": nt * bal["enthalpy_rise_W"] / 1e3,
            "heat_to_wall": nt * bal["Q_wall_out_W"] / 1e3, "closure": nt * bal["energy_closure_W"] / 1e3},
        "molar_conversion_bundle_mol_s": nt * (bal["F_in_mol_s"] - bal["F_out_mol_s"]),
        "dimensionless": {"k_tau_inlet": k_in * tau, "adiabatic_rise_K": dT_ad,
                          "arrhenius_gamma": E_J_mol / (R_GAS * inlet_T_K),
                          "radial_mass_number": diffusivity_m2_s * tau / R ** 2,
                          "radial_heat_number": thermal_conductivity_W_mK / rho_cp * tau / R ** 2,
                          "Pe_axial_mass": geo["u_mean_m_s"] * geo["tube_length_m"] / D_z,
                          "Pe_axial_heat": rho_cp * geo["u_mean_m_s"] * geo["tube_length_m"] / l_z},
        "mesh": {"coarse": c["mesh"], "fine": res["fine"]["mesh"] if "fine" in res else None},
        "coarse_mesh_result": {"X": c["X_outlet"], "T_outlet_K": c["T_outlet_cup_K"], "T_hot_K": c["hot_spot"]["T_max_K"]},
        "solver": {m: res[m]["solver"] | {"seconds": res[m]["seconds"]} for m in ("coarse", "fine") if m in res},
        "engine": f"FEniCSx dolfinx {res.get('dolfinx_version', '?')} (campo) + scipy Radau (PFR 1-D)",
        "evidence": ev, "assumptions": assumptions,
        "notes": ("El hot spot del campo es el máximo local (normalmente en el eje); el PFR 1-D solo ve la T de mezcla. "
                  "radial_heat_number = α·τ/R²: ≫ 1 → perfiles planos y el campo tiende al PFR; ≲ 1 → gradientes radiales y "
                  "el PFR subestima el punto caliente."),
    }
    if figure_path:
        fp = str(Path(figure_path).expanduser().resolve())
        save_figure(fp, f"T local (FEniCSx) · {velocity_profile} · pared {wall['mode']} · {n_tubes} tubos × "
                        f"{tube_diameter_m*100:.2f} cm × {geo['tube_length_m']:.2f} m", res, pfr)
        out["figure_path"] = fp
    return out
