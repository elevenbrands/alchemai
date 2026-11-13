"""IDAES generic-property package for an ideal mixture (Raoult + ideal gas), generated from `chemicals` / `thermo` data
(bloque M8b of docs/MAPA_MOTORES.md). No coefficient is typed by hand: each one is read from a published table that
`chemicals` ships, in an IDAES pure-component form that is the SAME equation, or fitted (liquid cp only) and checked.

  property            IDAES form           source in chemicals / thermo                      relation
  Psat                RPP4 (Wagner 3-6)    Psat_data_WagnerMcGarry (A-D, Tc, Pc)             same equation
  cp ideal gas        RPP5 (Cp/R a0-a4)    Cp_data_Poling (Poling et al. 2001, App. A)       same equation
  liquid density      Perrys eqn_type 1    rho_data_Perry_8E_105_l (DIPPR 105, mol/m³)       same equation (units converted)
  cp liquid           Perrys (DIPPR 100)   thermo HeatCapacityLiquid (its default method)    cubic least squares, residual reported
  ΔHvap anchor        H_ref vap − liq      thermo EnthalpyVaporization DIPPR_PERRY_8E (106)  exact at Tsat(P_column)
  H_form gas          enth_mol_form_vap    chemicals.reaction.Hfg                            value

In the ideal model ΔHvap(T) = H_ig(T) − H_liq(T) = ΔH_ref + ∫(cp_ig − cp_liq) dT. The liquid reference enthalpy is set so
that ΔHvap equals DIPPR 106 at each component's saturation temperature at the column pressure (where it boils in the
column); it is therefore an effective reference, not the liquid formation enthalpy. `property_check` compares the
package's Psat, cp, density and ΔHvap with the same-source tables (catches unit errors: must agree to ~1e-12) and with
independent DIPPR (Perry's 8th ed.) / TRC correlations over the column's temperature range.

The ideal assumption is screened with Dortmund-modified UNIFAC (thermo, DDBST group assignments): for each binary pair
at the column pressure, the bubble-point relative volatility with activity coefficients vs Raoult. An azeotrope or
max |ln(γi/γj)| > MAX_LN_GAMMA_RATIO (α off by more than ~10 %) refuses the mixture.
"""
from __future__ import annotations

import math

R = 8.314462618
T_REF = 298.15
MAX_LN_GAMMA_RATIO = 0.10      # relative-volatility error tolerated by the Raoult assumption
MAX_B_P_RT = 0.10              # |B·P/RT| (Tsonopoulos second virial) tolerated by the ideal-gas assumption
CP_LIQ_FIT_TOL = 0.02          # max relative residual of the liquid-cp cubic fit
PROPERTY_TOLS = {"psat_same_source": 1e-6, "psat_vs_dippr101": 0.03, "cp_ig_same_source": 1e-6, "cp_ig_vs_trc": 0.03,
                 "dens_liq_same_source": 1e-6, "cp_liq_vs_source": CP_LIQ_FIT_TOL, "dhvap_vs_dippr106": 0.03}
SOURCES = {
    "pressure_sat": "Wagner (forma 3-6) de McGarry 1983 vía chemicals.vapor_pressure.Psat_data_WagnerMcGarry → IDAES RPP4",
    "cp_ig": "Polinomio de Poling, Prausnitz y O'Connell (2001, apéndice A) vía chemicals.heat_capacity.Cp_data_Poling → IDAES RPP5",
    "dens_liq": "DIPPR 105 de Perry's 8.ª ed. (tabla 2-32) vía chemicals.volume.rho_data_Perry_8E_105_l → IDAES Perrys",
    "cp_liq": "thermo.HeatCapacityLiquid (método por defecto, nombrado por componente) ajustado a DIPPR 100 cúbico → IDAES Perrys",
    "dhvap_ref": "DIPPR 106 de Perry's 8.ª ed. vía thermo.EnthalpyVaporization(DIPPR_PERRY_8E); ΔHvap del paquete = DIPPR "
                 "en Tsat de cada compuesto a la presión de la columna",
    "h_form_gas": "chemicals.reaction.Hfg (entalpía de formación del gas ideal, 298.15 K)",
    "ideality": "UNIFAC modificado Dortmund (thermo.unifac, asignación de grupos DDBST)",
    "virial": "Segundo coeficiente virial de Tsonopoulos (chemicals.virial)",
}


def resolve(name: str) -> dict:
    from chemicals import CAS_from_any, MW
    from chemicals.identifiers import search_chemical
    try:
        cas = CAS_from_any(name)
    except Exception as ex:  # noqa: BLE001
        raise ValueError(f"compuesto no reconocido por chemicals: {name!r}") from ex
    meta = search_chemical(cas)
    return {"name": name, "CAS": cas, "common_name": meta.common_name, "formula": meta.formula, "MW_g_mol": MW(cas)}


def _f(x) -> float:
    return float(x)


def pure_data(name: str) -> dict:
    """Coefficients from the chemicals tables (raises if a table lacks the compound)."""
    import chemicals.vapor_pressure as VP
    import chemicals.heat_capacity as HC
    import chemicals.volume as V
    from chemicals.reaction import Hfg
    from thermo import EnthalpyVaporization, HeatCapacityLiquid
    info = resolve(name)
    cas = info["CAS"]
    missing = []
    if cas not in VP.Psat_data_WagnerMcGarry.index:
        missing.append("Psat (Wagner McGarry)")
    if cas not in HC.Cp_data_Poling.index or math.isnan(_f(HC.Cp_data_Poling.at[cas, "a0"])):
        missing.append("cp gas ideal (Poling)")
    if cas not in V.rho_data_Perry_8E_105_l.index:
        missing.append("densidad líquida (DIPPR 105)")
    dh = EnthalpyVaporization(CASRN=cas)
    if "DIPPR_PERRY_8E" not in dh.all_methods:
        missing.append("ΔHvap (DIPPR 106)")
    cpl = HeatCapacityLiquid(CASRN=cas)
    if cpl.method is None:
        missing.append("cp líquido")
    if missing:
        raise ValueError(f"{name}: faltan datos verificados en chemicals/thermo para {', '.join(missing)}; "
                         "no se generan coeficientes a mano")
    w = VP.Psat_data_WagnerMcGarry.loc[cas]
    p = HC.Cp_data_Poling.loc[cas]
    d = V.rho_data_Perry_8E_105_l.loc[cas]
    dh.method = "DIPPR_PERRY_8E"
    hf = Hfg(cas)
    return {**info,
            "wagner": {k: _f(w[k]) for k in ("A", "B", "C", "D")}, "Tc_K": _f(w["Tc"]), "Pc_Pa": _f(w["Pc"]),
            "psat_Tmin_K": _f(w["Tmin"]),
            "poling": {f"a{i}": _f(p[f"a{i}"]) for i in range(5)}, "cp_ig_range_K": (_f(p["Tmin"]), _f(p["Tmax"])),
            "dippr105": {k: _f(d[k]) for k in ("C1", "C2", "C3", "C4")}, "dens_range_K": (_f(d["Tmin"]), _f(d["Tmax"])),
            "dhvap_ref_J_mol": _f(dh(T_REF)), "dhvap_anchor": dh, "h_form_gas_J_mol": None if hf is None else _f(hf),
            "cp_liq_method": cpl.method}


def saturation_T(name: str, P_Pa: float) -> float:
    from thermo import VaporPressure
    from chemicals import CAS_from_any
    return float(VaporPressure(CASRN=CAS_from_any(name)).solve_property(P_Pa))


def fit_cp_liq(name: str, T_lo: float, T_hi: float) -> dict:
    """DIPPR 100 cubic (J/mol·K, T in K) least-squares fit to thermo's liquid cp over [T_lo, T_hi]."""
    import numpy as np
    from thermo import HeatCapacityLiquid
    from chemicals import CAS_from_any
    cpl = HeatCapacityLiquid(CASRN=CAS_from_any(name))
    Ts = np.linspace(T_lo, T_hi, 41)
    cp = np.array([cpl(T) for T in Ts], dtype=float)
    if not np.all(np.isfinite(cp)):
        raise ValueError(f"{name}: thermo no da cp líquido en {T_lo:.0f}-{T_hi:.0f} K ({cpl.method})")
    c3, c2, c1, c0 = np.polyfit(Ts, cp, 3)
    resid = float(np.max(np.abs(np.polyval([c3, c2, c1, c0], Ts) / cp - 1)))
    if resid > CP_LIQ_FIT_TOL:
        raise ValueError(f"{name}: el ajuste cúbico del cp líquido se desvía {resid:.1%} de {cpl.method}")
    return {"coeffs_J_mol_K": [float(c0), float(c1), float(c2), float(c3)], "method": cpl.method,
            "fit_range_K": (T_lo, T_hi), "max_rel_residual": resid}


def temperature_window(names: list[str], P_Pa: float, extra_T: list[float] = ()) -> tuple[float, float]:
    ts = [saturation_T(n, P_Pa) for n in names] + list(extra_T)
    return min(ts) - 30.0, max(ts) + 30.0


def ideality_check(names: list[str], P_Pa: float) -> dict:
    """Dortmund UNIFAC vs Raoult for every binary pair at P: bubble-point α over x ∈ [0, 1]."""
    import itertools
    import numpy as np
    from thermo.unifac import UNIFAC, UNIFAC_group_assignment_DDBST, DOUFSG, DOUFIP2016
    from thermo import VaporPressure
    from chemicals import CAS_from_any
    from chemicals.virial import BVirial_Tsonopoulos
    from chemicals import Tc as Tc_, Pc as Pc_, omega as omega_
    pairs, reasons = [], []
    groups, psat = {}, {}
    for n in names:
        cas = CAS_from_any(n)
        g = UNIFAC_group_assignment_DDBST(cas, "MODIFIED_UNIFAC")
        if not g:
            reasons.append(f"{n}: sin asignación de grupos UNIFAC; no se puede comprobar que la mezcla es ideal")
        groups[n], psat[n] = g, VaporPressure(CASRN=cas)
    if reasons:
        return {"ideal": False, "pairs": [], "reasons": reasons, "method": SOURCES["ideality"]}
    for a, b in itertools.combinations(names, 2):
        Ta, Tb = psat[a].solve_property(P_Pa), psat[b].solve_property(P_Pa)
        alphas, lnr, g_inf = [], [], {}
        for x in np.linspace(0.0, 1.0, 21):
            xs = [float(x), float(1 - x)]
            lo, hi = min(Ta, Tb) - 40, max(Ta, Tb) + 40
            for _ in range(60):
                T = 0.5 * (lo + hi)
                G = UNIFAC.from_subgroups(T, xs, [groups[a], groups[b]], version=1, interaction_data=DOUFIP2016,
                                          subgroups=DOUFSG).gammas()
                p = xs[0] * G[0] * psat[a](T) + xs[1] * G[1] * psat[b](T)
                lo, hi = (T, hi) if p < P_Pa else (lo, T)
            alphas.append(G[0] * psat[a](T) / (G[1] * psat[b](T)))
            lnr.append(abs(math.log(G[0] / G[1])))
            if x == 0.0:
                g_inf[a] = G[0]
            if x == 1.0:
                g_inf[b] = G[1]
        azeo = (min(alphas) - 1) * (max(alphas) - 1) < 0
        dev = max(lnr)
        pairs.append({"pair": [a, b], "gamma_inf": g_inf, "alpha_min": min(alphas), "alpha_max": max(alphas),
                      "max_ln_gamma_ratio": dev, "azeotrope_predicted": azeo})
        if azeo:
            reasons.append(f"{a}/{b}: UNIFAC predice un azeótropo a {P_Pa:.0f} Pa (α cruza 1); Raoult no sirve")
        elif dev > MAX_LN_GAMMA_RATIO:
            reasons.append(f"{a}/{b}: no ideal (γ∞ {g_inf[a]:.2f} / {g_inf[b]:.2f}; α con Raoult se equivoca hasta "
                           f"{math.exp(dev) - 1:.0%})")
    virial = {}
    for n in names:
        cas = CAS_from_any(n)
        T = psat[n].solve_property(P_Pa)
        B = BVirial_Tsonopoulos(T, Tc_(cas), Pc_(cas), omega_(cas))
        virial[n] = B * P_Pa / (R * T)
        if abs(virial[n]) > MAX_B_P_RT:
            reasons.append(f"{n}: el vapor no es gas ideal a {P_Pa:.0f} Pa (|B·P/RT| = {abs(virial[n]):.2f})")
    return {"ideal": not reasons, "pairs": pairs, "B_P_over_RT_at_Tsat": virial, "reasons": reasons,
            "tolerances": {"max_ln_gamma_ratio": MAX_LN_GAMMA_RATIO, "max_B_P_RT": MAX_B_P_RT},
            "method": SOURCES["ideality"] + "; " + SOURCES["virial"]}


def build_configuration(names: list[str], P_Pa: float, extra_T: list[float] = ()) -> tuple[dict, dict]:
    """(IDAES configuration, provenance). Component keys are the names given by the user."""
    from pyomo.environ import units as pyunits
    from idaes.core import LiquidPhase, VaporPhase, Component
    from idaes.models.properties.modular_properties.state_definitions import FTPx
    from idaes.models.properties.modular_properties.eos.ideal import Ideal
    from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
    from idaes.models.properties.modular_properties.phase_equil.bubble_dew import IdealBubbleDew
    from idaes.models.properties.modular_properties.phase_equil.forms import fugacity
    from idaes.models.properties.modular_properties.pure import Perrys, RPP4, RPP5
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("se necesitan 2 o más compuestos distintos")
    T_lo, T_hi = temperature_window(names, P_Pa, extra_T)
    comps, prov = {}, {"temperature_window_K": (T_lo, T_hi), "T_ref_K": T_REF, "sources": SOURCES, "components": {}}
    J, mol, K = pyunits.J, pyunits.mol, pyunits.K
    for n in names:
        d = pure_data(n)
        if T_hi >= d["Tc_K"]:
            raise ValueError(f"{n}: la ventana de temperatura llega a {T_hi:.0f} K ≥ Tc {d['Tc_K']:.0f} K")
        cpl = fit_cp_liq(n, T_lo, T_hi)
        c = cpl["coeffs_J_mol_K"]
        T_a = min(max(saturation_T(n, P_Pa), T_lo), T_hi)
        a = [d["poling"][f"a{i}"] for i in range(5)]
        kirchhoff = (sum(R * a[i] / (i + 1) * (T_a ** (i + 1) - T_REF ** (i + 1)) for i in range(5))
                     - sum(c[i] / (i + 1) * (T_a ** (i + 1) - T_REF ** (i + 1)) for i in range(4)))
        dhv = d["dhvap_anchor"](T_a) - kirchhoff          # ΔH_ref that makes ΔHvap(T_a) = DIPPR 106
        hfg = d["h_form_gas_J_mol"] if d["h_form_gas_J_mol"] is not None else dhv
        comps[n] = {
            "type": Component,
            "dens_mol_liq_comp": Perrys, "enth_mol_liq_comp": Perrys, "enth_mol_ig_comp": RPP5,
            "pressure_sat_comp": RPP4, "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
            "parameter_data": {
                "mw": (d["MW_g_mol"] * 1e-3, pyunits.kg / mol),
                "pressure_crit": (d["Pc_Pa"], pyunits.Pa), "temperature_crit": (d["Tc_K"], K),
                "dens_mol_liq_comp_coeff": {"eqn_type": 1, "1": (d["dippr105"]["C1"], mol * pyunits.m ** -3),
                                            "2": (d["dippr105"]["C2"], None), "3": (d["dippr105"]["C3"], K),
                                            "4": (d["dippr105"]["C4"], None)},
                "cp_mol_ig_comp_coeff": {"a0": (d["poling"]["a0"], None), "a1": (d["poling"]["a1"], K ** -1),
                                         "a2": (d["poling"]["a2"], K ** -2), "a3": (d["poling"]["a3"], K ** -3),
                                         "a4": (d["poling"]["a4"], K ** -4)},
                "cp_mol_liq_comp_coeff": {"1": (c[0], J / mol / K), "2": (c[1], J / mol / K ** 2),
                                          "3": (c[2], J / mol / K ** 3), "4": (c[3], J / mol / K ** 4),
                                          "5": (0.0, J / mol / K ** 5)},
                "enth_mol_form_vap_comp_ref": (hfg, J / mol),
                "enth_mol_form_liq_comp_ref": (hfg - dhv, J / mol),
                "pressure_sat_comp_coeff": {k: (d["wagner"][k], None) for k in ("A", "B", "C", "D")},
            },
        }
        ranges = {"psat_from_K": d["psat_Tmin_K"], "cp_ig_K": d["cp_ig_range_K"], "dens_liq_K": d["dens_range_K"]}
        extrap = [k for k, v in ranges.items() if (T_lo < v if isinstance(v, float) else (T_lo < v[0] or T_hi > v[1]))]
        prov["components"][n] = {"CAS": d["CAS"], "common_name": d["common_name"], "formula": d["formula"],
                                 "MW_g_mol": d["MW_g_mol"], "Tc_K": d["Tc_K"], "Pc_Pa": d["Pc_Pa"],
                                 "dhvap_dippr106_298_kJ_mol": d["dhvap_ref_J_mol"] / 1e3,
                                 "dhvap_anchor_T_K": T_a, "dhvap_anchor_kJ_mol": d["dhvap_anchor"](T_a) / 1e3,
                                 "h_ref_liq_kJ_mol_effective": (hfg - dhv) / 1e3,
                                 "h_form_gas_kJ_mol": None if d["h_form_gas_J_mol"] is None else hfg / 1e3,
                                 "cp_liq_fit": cpl, "correlation_ranges_K": ranges,
                                 "window_outside_correlation_range": extrap}
    cfg = {
        "components": comps,
        "phases": {"Liq": {"type": LiquidPhase, "equation_of_state": Ideal},
                   "Vap": {"type": VaporPhase, "equation_of_state": Ideal}},
        "base_units": {"time": pyunits.s, "length": pyunits.m, "mass": pyunits.kg, "amount": pyunits.mol,
                       "temperature": pyunits.K},
        "state_definition": FTPx,
        "state_bounds": {"flow_mol": (0, 100, 1e5, mol / pyunits.s),
                         "temperature": (T_lo, 0.5 * (T_lo + T_hi), T_hi, K),
                         "pressure": (0.5 * P_Pa, P_Pa, 2.0 * P_Pa, pyunits.Pa)},
        "pressure_ref": (1e5, pyunits.Pa),
        "temperature_ref": (T_REF, K),
        "phases_in_equilibrium": [("Vap", "Liq")],
        "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
        "bubble_dew_method": IdealBubbleDew,
    }
    return cfg, prov


def property_check(cfg: dict, names: list[str], temperatures: list[float]) -> dict:
    """Values computed BY THE IDAES PACKAGE vs chemicals/thermo at each T (same-source checks catch unit errors; DIPPR
    checks are an independent second source)."""
    from pyomo.environ import ConcreteModel, value, units as pyunits
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    from thermo import VaporPressure, HeatCapacityGas, HeatCapacityLiquid, EnthalpyVaporization, VolumeLiquid
    from chemicals import CAS_from_any
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.p = GenericParameterBlock(**cfg)
    m.fs.sb = m.fs.p.build_state_block([0], defined_state=True)
    s = m.fs.sb[0]
    s.flow_mol.fix(1)
    s.pressure.fix(101325)
    for n in names:
        s.mole_frac_comp[n].fix(1.0 / len(names))
    rows, worst = [], {k: 0.0 for k in PROPERTY_TOLS}
    for n in names:
        cas = CAS_from_any(n)
        ps_w, ps_d = VaporPressure(CASRN=cas), VaporPressure(CASRN=cas)
        ps_w.method, ps_d.method = "WAGNER_MCGARRY", "DIPPR_PERRY_8E"
        cg_p, cg_t = HeatCapacityGas(CASRN=cas), HeatCapacityGas(CASRN=cas)
        cg_p.method = "POLING_POLY"
        cg_t.method = "TRCIG" if "TRCIG" in cg_t.all_methods else cg_t.method
        cl = HeatCapacityLiquid(CASRN=cas)
        dh = EnthalpyVaporization(CASRN=cas)
        dh.method = "DIPPR_PERRY_8E"
        vl = VolumeLiquid(CASRN=cas)
        vl.method = "DIPPR_PERRY_8E"
        cobj = m.fs.p.get_component(n)
        for T in temperatures:
            s.temperature.fix(T)
            spec = cfg["components"][n]
            T_ = s.temperature
            psat = value(pyunits.convert(spec["pressure_sat_comp"].pressure_sat_comp.return_expression(s, cobj, T_),
                                         pyunits.Pa))
            cpig = value(pyunits.convert(spec["enth_mol_ig_comp"].cp_mol_ig_comp.return_expression(s, cobj, T_),
                                         pyunits.J / pyunits.mol / pyunits.K))
            cpl = value(pyunits.convert(spec["enth_mol_liq_comp"].cp_mol_liq_comp.return_expression(s, cobj, T_),
                                        pyunits.J / pyunits.mol / pyunits.K))
            rho = value(pyunits.convert(spec["dens_mol_liq_comp"].dens_mol_liq_comp.return_expression(s, cobj, T_),
                                        pyunits.mol / pyunits.m ** 3))
            dhv = value(s.enth_mol_phase_comp["Vap", n]) - value(s.enth_mol_phase_comp["Liq", n])
            r = {"component": n, "T_K": T,
                 "psat_Pa": psat, "psat_wagner_mcgarry_Pa": ps_w(T), "psat_dippr101_Pa": ps_d(T),
                 "cp_ig_J_mol_K": cpig, "cp_ig_poling": cg_p(T), "cp_ig_" + cg_t.method.lower(): cg_t(T),
                 "cp_liq_J_mol_K": cpl, "cp_liq_" + cl.method.lower(): cl(T),
                 "dens_liq_mol_m3": rho, "dens_liq_dippr105_mol_m3": 1.0 / vl.T_dependent_property(T),
                 "dhvap_kJ_mol": dhv / 1e3, "dhvap_dippr106_kJ_mol": dh(T) / 1e3}
            rows.append(r)
            for k, got, ref in (("psat_same_source", psat, ps_w(T)), ("psat_vs_dippr101", psat, ps_d(T)),
                                ("cp_ig_same_source", cpig, cg_p(T)), ("cp_ig_vs_trc", cpig, cg_t(T)),
                                ("dens_liq_same_source", rho, 1.0 / vl.T_dependent_property(T)), ("cp_liq_vs_source", cpl, cl(T)),
                                ("dhvap_vs_dippr106", dhv, dh(T))):
                worst[k] = max(worst[k], abs(got / ref - 1))
    ok = {k: worst[k] <= tol for k, tol in PROPERTY_TOLS.items()}
    return {"rows": rows, "max_rel_diff": worst, "tolerances": PROPERTY_TOLS, "within_tolerance": ok,
            "verified": all(ok.values())}
