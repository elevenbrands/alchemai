"""Shell side, boiling and condensation (bloque M4b): Kern's shell-side method, nucleate pool boiling with critical
heat flux, and film condensation (Nusselt outside, Shah / Boyko-Kruzhilin inside tubes).

The correlations come from Caleb Bell's MIT-licensed `ht`. The closed forms in `evidence` are written here from the
printed equations (Kern 1950; Cooper 1984; Rohsenow 1952 and Zuber/Lienhard-Dhir as in Incropera ch. 10; Nusselt
1916 with Rohsenow's modified latent heat). `evidence.verified` means the correlation was applied inside its
validity range with consistent geometry and reproduces the printed equation; it does not make the correlation
more accurate than its own scatter, which the notes state.
"""
from __future__ import annotations

import math

G = 9.80665


# ---------------------------------------------------------------- shell side, Kern (1950)
def _kern_De(pitch: float, Do: float, layout: str) -> float:
    if layout == "square":
        return 4.0 * (pitch ** 2 - math.pi * Do ** 2 / 4.0) / (math.pi * Do)
    if layout == "triangular":
        return 4.0 * (math.sqrt(3.0) / 4.0 * pitch ** 2 - math.pi * Do ** 2 / 8.0) / (math.pi * Do / 2.0)
    raise ValueError("layout must be 'triangular' or 'square'")


def shell_kern(mass_flow_kg_s: float, density_kg_m3: float, viscosity_Pa_s: float, thermal_conductivity_W_mK: float,
               cp_J_kgK: float, shell_diameter_m: float, baffle_spacing_m: float, n_baffles: int, tube_od_m: float,
               tube_pitch_m: float, layout: str = "triangular", wall_viscosity_Pa_s: float | None = None) -> dict:
    from ht.conv_tube_bank import Nu_Zukauskas_Bejan, dP_Kern
    m, rho, mu, k, cp = mass_flow_kg_s, density_kg_m3, viscosity_Pa_s, thermal_conductivity_W_mK, cp_J_kgK
    Ds, B, Do, Pt = shell_diameter_m, baffle_spacing_m, tube_od_m, tube_pitch_m
    De = _kern_De(Pt, Do, layout)
    As = Ds * (Pt - Do) * B / Pt                         # crossflow area at the shell centerline
    Gs = m / As
    Re, Pr = De * Gs / mu, cp * mu / k
    visc = (mu / wall_viscosity_Pa_s) ** 0.14 if wall_viscosity_Pa_s else 1.0
    Nu = 0.36 * Re ** 0.55 * Pr ** (1.0 / 3.0) * visc     # Kern (1950), valid 2e3 < Re < 1e6
    h = Nu * k / De
    dP = float(dP_Kern(m=m, rho=rho, mu=mu, DShell=Ds, LSpacing=B, pitch=Pt, Do=Do, NBaffles=n_baffles,
                       mu_w=wall_viscosity_Pa_s))
    # ideal tube bank at the same maximum velocity (Žukauskas, as in Bejan): Kern sits below it because the real
    # shell has leakage and bypass streams that the ideal bank does not
    Vmax = Gs / rho
    rows = max(1, round(Ds / (Pt * (math.sqrt(3.0) / 2.0 if layout == "triangular" else 1.0))))
    SL = Pt * math.sqrt(3.0) / 2.0 if layout == "triangular" else Pt
    ST = Pt
    Re_bank = rho * Vmax * Do / mu
    h_bank = float(Nu_Zukauskas_Bejan(Re=Re_bank, Pr=Pr, tube_rows=rows, pitch_parallel=SL, pitch_normal=ST)) * k / Do
    reasons, warnings = [], []
    if not 2e3 <= Re <= 1e6:
        reasons.append(f"Re = {Re:.3g} fuera del rango de Kern (2·10³-10⁶)")
    if Pt / Do < 1.25 - 1e-9:
        reasons.append(f"paso/diámetro = {Pt / Do:.3f} < 1.25 (mínimo TEMA)")
    B_min = max(Ds / 5.0, 0.0508)
    if B < B_min - 1e-9 or B > Ds + 1e-9:
        reasons.append(f"espaciado de mamparas {B:.3f} m fuera de [máx(Ds/5, 2 in), Ds] = [{B_min:.3f}, {Ds:.3f}] m")
    if wall_viscosity_Pa_s is None:
        warnings.append("sin viscosidad a la temperatura de pared: (μ/μw)^0.14 = 1 (aceptable para gases y "
                        "líquidos poco viscosos)")
    Nu_chk = 0.36 * (De * m / As / mu) ** 0.55 * (cp * mu / k) ** (1 / 3) * visc
    match = abs(Nu_chk - Nu) / Nu < 1e-12
    return {"h_W_m2K": h, "dP_Pa": dP, "Re": Re, "Pr": Pr, "Nu": Nu, "equivalent_diameter_m": De,
            "crossflow_area_m2": As, "mass_velocity_kg_m2_s": Gs, "velocity_m_s": Vmax, "viscosity_correction": visc,
            "h_ideal_tube_bank_W_m2K": h_bank, "kern_over_ideal_bank": h / h_bank, "warnings": warnings,
            "evidence": {"formula_match": match, "verified": match and not reasons, "reasons": reasons},
            "notes": ("Kern no modela las fugas mampara-coraza y tubo-mampara ni el bypass del haz: es un método de "
                      "estimación (típicamente conservador en h y poco fiable en ΔP). Para diseño final se usa "
                      "Bell-Delaware (pendiente, bloque M4c). Compárese con h_ideal_tube_bank_W_m2K, que es el techo "
                      "sin fugas.")}


# ---------------------------------------------------------------- nucleate pool boiling + critical heat flux
def _cooper(P: float, Pc: float, MW: float, q: float, Rp_um: float) -> float:
    pr = P / Pc
    return 55.0 * pr ** (0.12 - 0.2 * math.log10(Rp_um)) * (-math.log10(pr)) ** -0.55 * MW ** -0.5 * q ** 0.67


def pool_boiling(P_Pa: float, critical_pressure_Pa: float, MW: float, T_sat_K: float, liquid_density_kg_m3: float,
                 vapor_density_kg_m3: float, liquid_viscosity_Pa_s: float, liquid_conductivity_W_mK: float,
                 liquid_cp_J_kgK: float, latent_heat_J_kg: float, surface_tension_N_m: float,
                 heat_flux_W_m2: float | None = None, wall_superheat_K: float | None = None,
                 surface_roughness_um: float = 1.0, rohsenow_Csf: float | None = None, rohsenow_n: float | None = None) -> dict:
    from ht.boiling_nucleic import Cooper, Montinsky, Rohsenow, Stephan_Abdelsalam
    if (heat_flux_W_m2 is None) == (wall_superheat_K is None):
        raise ValueError("give exactly one of heat_flux_W_m2 or wall_superheat_K")
    rl, rg, mul, kl, cpl = liquid_density_kg_m3, vapor_density_kg_m3, liquid_viscosity_Pa_s, liquid_conductivity_W_mK, liquid_cp_J_kgK
    hfg, sigma = latent_heat_J_kg, surface_tension_N_m
    kw = {"q": heat_flux_W_m2} if heat_flux_W_m2 is not None else {"Te": wall_superheat_K}
    h = {"Cooper": float(Cooper(P=P_Pa, Pc=critical_pressure_Pa, MW=MW, Rp=surface_roughness_um * 1e-6, **kw)),
         "Mostinski": float(Montinsky(P=P_Pa, Pc=critical_pressure_Pa, **kw)),
         "Stephan_Abdelsalam": float(Stephan_Abdelsalam(rhol=rl, rhog=rg, mul=mul, kl=kl, Cpl=cpl, Hvap=hfg, sigma=sigma,
                                                         Tsat=T_sat_K, **kw))}
    if rohsenow_Csf is not None:
        if rohsenow_n is None:
            raise ValueError("Rohsenow needs both rohsenow_Csf and rohsenow_n (surface-fluid pair, e.g. water: n = 1)")
        h["Rohsenow"] = float(Rohsenow(rhol=rl, rhog=rg, mul=mul, kl=kl, Cpl=cpl, Hvap=hfg, sigma=sigma,
                                       Csf=rohsenow_Csf, n=rohsenow_n, **kw))
    rec = "Cooper"
    q = heat_flux_W_m2 if heat_flux_W_m2 is not None else h[rec] * wall_superheat_K
    Te = wall_superheat_K if wall_superheat_K is not None else heat_flux_W_m2 / h[rec]
    base = sigma * G * (rl - rg)
    qmax = {"Zuber_K0.131": 0.131 * hfg * math.sqrt(rg) * base ** 0.25,
            "Lienhard_Dhir_K0.149": 0.149 * hfg * math.sqrt(rg) * base ** 0.25}
    margin = q / qmax["Zuber_K0.131"]
    # independent closed forms
    if heat_flux_W_m2 is not None:
        cooper_chk = _cooper(P_Pa, critical_pressure_Pa, MW, q, surface_roughness_um)
    else:  # h = 55·f(pr)·M^-0.5·(h·ΔT)^0.67 → h^0.33 = 55·f·M^-0.5·ΔT^0.67
        cooper_chk = (_cooper(P_Pa, critical_pressure_Pa, MW, 1.0, surface_roughness_um) * Te ** 0.67) ** (1.0 / 0.33)
    match = abs(cooper_chk - h["Cooper"]) / h["Cooper"] < 1e-9
    if "Rohsenow" in h:
        Pr = cpl * mul / kl
        qR = mul * hfg * math.sqrt(G * (rl - rg) / sigma) * (cpl * Te / (rohsenow_Csf * hfg * Pr ** rohsenow_n)) ** 3
        if wall_superheat_K is not None:
            match = match and abs(qR / wall_superheat_K - h["Rohsenow"]) / h["Rohsenow"] < 1e-9
    pr = P_Pa / critical_pressure_Pa
    reasons = [] if match else ["la correlación no coincide con su ecuación impresa"]
    if not 0.001 <= pr <= 0.9:
        reasons.append(f"presión reducida {pr:.4f} fuera de 0.001-0.9 (rango de Cooper)")
    if margin >= 1.0:
        reasons.append(f"q = {q:.3g} W/m² supera el flujo crítico de Zuber {qmax['Zuber_K0.131']:.3g}: ebullición "
                       "de película, el h nucleado no aplica")
    vals = list(h.values())
    return {"h_W_m2K": h[rec], "recommended_method": rec, "h_by_method_W_m2K": h, "heat_flux_W_m2": q,
            "wall_superheat_K": Te, "q_max_W_m2": qmax, "fraction_of_critical_heat_flux": margin,
            "method_spread_pct": (max(vals) - min(vals)) / h[rec] * 100.0, "reduced_pressure": pr,
            "evidence": {"cooper_closed_form_W_m2K": cooper_chk, "formula_match": match, "verified": not reasons,
                         "reasons": reasons},
            "notes": ("Ebullición nucleada en alberca sobre una superficie (Cooper 1984, rugosidad Rp en μm; "
                      "recomendada por depender solo de P/Pc y M). Las correlaciones dispersan ±30-50 %: reportar "
                      "el intervalo. En haces de tubos (rehervidor tipo kettle) el flujo crítico baja respecto al de "
                      "una superficie aislada; diseñar lejos de q_max.")}


# ---------------------------------------------------------------- film condensation
def condensation(geometry: str, T_sat_K: float, liquid_density_kg_m3: float, vapor_density_kg_m3: float,
                 liquid_conductivity_W_mK: float, liquid_viscosity_Pa_s: float, liquid_cp_J_kgK: float,
                 latent_heat_J_kg: float, wall_T_K: float | None = None, length_m: float | None = None,
                 tube_od_m: float | None = None, n_rows: int = 1, mass_flow_kg_s: float | None = None,
                 quality: float | None = None, tube_id_m: float | None = None, P_Pa: float | None = None,
                 critical_pressure_Pa: float | None = None) -> dict:
    rl, rg, kl, mul, cpl, hfg = (liquid_density_kg_m3, vapor_density_kg_m3, liquid_conductivity_W_mK,
                                 liquid_viscosity_Pa_s, liquid_cp_J_kgK, latent_heat_J_kg)
    if geometry == "in_tube":
        from ht.condensation import Boyko_Kruzhilin, Shah
        if None in (mass_flow_kg_s, quality, tube_id_m, P_Pa, critical_pressure_Pa):
            raise ValueError("in_tube needs mass_flow_kg_s (per tube), quality, tube_id_m, P_Pa, critical_pressure_Pa")
        h = {"Shah": float(Shah(m=mass_flow_kg_s, x=quality, D=tube_id_m, rhol=rl, mul=mul, kl=kl, Cpl=cpl, P=P_Pa,
                                Pc=critical_pressure_Pa)),
             "Boyko_Kruzhilin": float(Boyko_Kruzhilin(m=mass_flow_kg_s, rhog=rg, rhol=rl, kl=kl, mul=mul, Cpl=cpl,
                                                      D=tube_id_m, x=quality))}
        # Shah (1979): h = h_LO·[(1 − x)^0.8 + 3.8·x^0.76·(1 − x)^0.04 / pr^0.38], h_LO by Dittus-Boelter (all liquid)
        A = math.pi * tube_id_m ** 2 / 4.0
        Re_lo, Pr = mass_flow_kg_s / A * tube_id_m / mul, cpl * mul / kl
        h_lo = 0.023 * Re_lo ** 0.8 * Pr ** 0.4 * kl / tube_id_m
        pr = P_Pa / critical_pressure_Pa
        x = quality
        chk = h_lo * ((1 - x) ** 0.8 + 3.8 * x ** 0.76 * (1 - x) ** 0.04 / pr ** 0.38)
        match = abs(chk - h["Shah"]) / h["Shah"] < 1e-6
        reasons = [] if match else ["Shah no coincide con su ecuación impresa"]
        if Re_lo < 350:
            reasons.append(f"Re_LO = {Re_lo:.0f} < 350 (fuera del rango de Shah)")
        if not 0.0 < x < 1.0:
            reasons.append("la calidad debe estar entre 0 y 1")
        vals = list(h.values())
        return {"geometry": geometry, "h_W_m2K": h["Shah"], "recommended_method": "Shah", "h_by_method_W_m2K": h,
                "method_spread_pct": (max(vals) - min(vals)) / h["Shah"] * 100.0, "Re_liquid_only": Re_lo,
                "evidence": {"shah_closed_form_W_m2K": chk, "formula_match": match, "verified": not reasons,
                             "reasons": reasons},
                "notes": "h local a esa calidad: para el condensador completo integrar a lo largo de x (1 → 0)."}
    if wall_T_K is None or wall_T_K >= T_sat_K:
        raise ValueError("film condensation needs wall_T_K below T_sat_K")
    dT = T_sat_K - wall_T_K
    Ja = cpl * dT / hfg
    hfg_mod = hfg * (1.0 + 0.68 * Ja)                     # Rohsenow's modified latent heat (subcooled film)
    group = G * rl * (rl - rg) * kl ** 3 * hfg_mod / mul
    if geometry in ("vertical_plate", "vertical_tube"):
        if length_m is None:
            raise ValueError("vertical surface needs length_m (height)")
        from ht.condensation import Nusselt_laminar
        h_val = float(Nusselt_laminar(Tsat=T_sat_K, Tw=wall_T_K, rhog=rg, rhol=rl, kl=kl, mul=mul, Hvap=hfg_mod,
                                      L=length_m))
        chk = 2.0 * math.sqrt(2.0) / 3.0 * (group / (dT * length_m)) ** 0.25    # 0.943
        Re_film = 4.0 * h_val * length_m * dT / (mul * hfg_mod)        # 4Γ/μ at the bottom of the surface
        size, form = length_m, "Nusselt vertical: h = 0.943·[g·ρl(ρl−ρv)·k³·h'fg/(μ·ΔT·L)]^¼"
    elif geometry == "horizontal_tube":
        if tube_od_m is None:
            raise ValueError("horizontal tube needs tube_od_m")
        h_val = 0.729 * (group / (dT * tube_od_m)) ** 0.25 * n_rows ** -0.25
        chk = 0.729 * (group / (dT * tube_od_m * n_rows)) ** 0.25
        Re_film = 4.0 * h_val * math.pi * tube_od_m / 2.0 * dT / (mul * hfg_mod) * n_rows   # per unit length, 2 sides
        size, form = tube_od_m, "Nusselt tubo horizontal: h = 0.729·[g·ρl(ρl−ρv)·k³·h'fg/(μ·ΔT·N·D)]^¼"
    else:
        raise ValueError("geometry must be 'vertical_plate', 'vertical_tube', 'horizontal_tube' or 'in_tube'")
    match = abs(chk - h_val) / h_val < 1e-9
    reasons = [] if match else ["no coincide con la ecuación de Nusselt impresa"]
    regime = "laminar sin ondas" if Re_film < 30 else ("laminar ondulado" if Re_film < 1800 else "turbulento")
    limit = 30.0 if geometry != "horizontal_tube" else 1800.0     # short path around a tube: ripples matter little
    if Re_film >= limit:
        reasons.append(f"Re de película = {Re_film:.0f} ≥ {limit:.0f} ({regime}): Nusselt subestima h; usar una correlación de "
                       "película ondulada o turbulenta")
    if Ja > 0.1:
        reasons.append(f"Ja = {Ja:.3f} > 0.1: el subenfriamiento de la película ya no es pequeño")
    return {"geometry": geometry, "h_W_m2K": h_val, "heat_flux_W_m2": h_val * dT, "Jakob": Ja,
            "modified_latent_heat_J_kg": hfg_mod, "film_Reynolds": Re_film, "film_regime": regime,
            "characteristic_length_m": size, "rows": n_rows,
            "evidence": {"closed_form": form, "closed_form_W_m2K": chk, "formula_match": match, "verified": not reasons,
                         "reasons": reasons},
            "notes": ("Condensación en película de vapor puro saturado. Con incondensables o mezclas el h cae "
                      "mucho (resistencia de difusión): esta tool no la modela.")}
