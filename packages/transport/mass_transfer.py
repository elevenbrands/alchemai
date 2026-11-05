"""Mass transfer for gas absorption (bloque 22): equilibrium stages (Kremser), transfer units (Colburn), film
coefficients for random packing (Onda, Takeuchi & Okumoto 1968) and packed height Z = H_OG · N_OG; diffusivities by
Wilke-Chang (liquid, dilute) and Fuller-Schettler-Giddings (gas).

Sources checked for this module:
  Onda et al., J. Chem. Eng. Japan 1(1), 56-62 (1968), Eqs. (1)-(3), read from the original paper (J-STAGE):
    a_w/a_t = 1 − exp{−1.45 (σc/σ)^0.75 Re_L^0.1 Fr_L^−0.05 We_L^0.2},  Re_L = L/(a_t μL), Fr_L = L² a_t/(ρL² g),
             We_L = L²/(ρL σ a_t)   (L = superficial liquid MASS velocity, kg/m²s)
    kL (ρL/(μL g))^(1/3) = 0.0051 (L/(a_w μL))^(2/3) Sc_L^(−1/2) (a_t Dp)^0.4
    kG R T/(a_t DG) = 5.23 (G/(a_t μG))^0.7 Sc_G^(1/3) (a_t Dp)^−2.0   (2.00 instead of 5.23 for packings < 15 mm)
  Wilke-Chang and Fuller-Schettler-Giddings forms: Clarkson Univ. CH330 notes (Welty et al. Eqs. 24-42, 24-52).
Kremser and Colburn are the textbook closed forms for straight operating and equilibrium lines (dilute systems).
"""
from __future__ import annotations

import math

G = 9.80665
R = 8.314462618


def kremser_fraction_absorbed(A: float, N: float) -> float:
    """Fraction of the solute absorbed in N equilibrium stages, solute-free lean solvent: (A^(N+1) − A)/(A^(N+1) − 1)."""
    if abs(A - 1.0) < 1e-9:
        return N / (N + 1.0)
    return (A ** (N + 1) - A) / (A ** (N + 1) - 1.0)


def kremser_stages(A: float, y_in: float, y_out: float, m: float, x_in: float = 0.0) -> float:
    """Equilibrium stages to absorb from y_in (gas in, bottom) to y_out (gas out, top) with lean liquid x_in (top)."""
    r = (y_in - m * x_in) / (y_out - m * x_in)
    if abs(A - 1.0) < 1e-9:
        return r - 1.0
    if A < 1.0 and r >= 1.0 / (1.0 - A) - 1e-12:
        raise ValueError(f"A = {A:.3f} < 1 caps the fraction absorbed at {A:.3f}: target unreachable with any N")
    return math.log(r * (1.0 - 1.0 / A) + 1.0 / A) / math.log(A)


def colburn_NOG(y_in: float, y_out: float, m: float, A: float, x_in: float = 0.0) -> float:
    """Overall gas-phase transfer units (Colburn), straight lines: ln[(1 − 1/A)(y_in − m x_in)/(y_out − m x_in) + 1/A]/(1 − 1/A)."""
    r = (y_in - m * x_in) / (y_out - m * x_in)
    if abs(A - 1.0) < 1e-9:
        return r - 1.0
    S = 1.0 / A
    arg = (1.0 - S) * r + S
    if arg <= 0:
        raise ValueError("target unreachable: A too small for this recovery")
    return math.log(arg) / (1.0 - S)


def onda(L_kg_m2s: float, G_kg_m2s: float, rho_L: float, mu_L: float, sigma_L: float, D_L: float, rho_G: float,
         mu_G: float, D_G: float, T_K: float, a_t: float, Dp: float, sigma_c: float) -> dict:
    """Onda (1968) wetted area and film coefficients, SI in and out. Returns a_w (m²/m³), kL (m/s), kG_ms = kG·R·T (m/s)."""
    Re_L, Fr_L, We_L = L_kg_m2s / (a_t * mu_L), L_kg_m2s ** 2 * a_t / (rho_L ** 2 * G), L_kg_m2s ** 2 / (rho_L * sigma_L * a_t)
    a_w = a_t * (1.0 - math.exp(-1.45 * (sigma_c / sigma_L) ** 0.75 * Re_L ** 0.1 * Fr_L ** -0.05 * We_L ** 0.2))
    Sc_L, Sc_G = mu_L / (rho_L * D_L), mu_G / (rho_G * D_G)
    kL = 0.0051 * (L_kg_m2s / (a_w * mu_L)) ** (2.0 / 3.0) * Sc_L ** -0.5 * (a_t * Dp) ** 0.4 / (rho_L / (mu_L * G)) ** (1.0 / 3.0)
    C = 2.00 if Dp < 0.015 else 5.23
    kG_ms = C * a_t * D_G * (G_kg_m2s / (a_t * mu_G)) ** 0.7 * Sc_G ** (1.0 / 3.0) * (a_t * Dp) ** -2.0
    return {"a_w_m2_m3": a_w, "wetted_fraction": a_w / a_t, "kL_m_s": kL, "kG_m_s": kG_ms, "kG_mol_m2_s_Pa": kG_ms / (R * T_K),
            "constant_gas": C, "Re_L": Re_L, "Fr_L": Fr_L, "We_L": We_L, "Sc_L": Sc_L, "Sc_G": Sc_G}


def packed_height(G_mol_m2s: float, L_mol_m2s: float, m: float, a_w: float, kG_m_s: float, kL_m_s: float,
                  c_G_mol_m3: float, c_L_mol_m3: float, N_OG: float) -> dict:
    """H_G = Gm/(ky a_w), H_L = Lm/(kx a_w), H_OG = H_G + (m Gm/Lm) H_L, Z = H_OG · N_OG (dilute, two-film)."""
    ky, kx = kG_m_s * c_G_mol_m3, kL_m_s * c_L_mol_m3
    H_G, H_L = G_mol_m2s / (ky * a_w), L_mol_m2s / (kx * a_w)
    lam = m * G_mol_m2s / L_mol_m2s
    H_OG = H_G + lam * H_L
    return {"H_G_m": H_G, "H_L_m": H_L, "stripping_factor": lam, "H_OG_m": H_OG, "N_OG": N_OG, "height_m": H_OG * N_OG,
            "gas_film_resistance_fraction": H_G / H_OG}


def wilke_chang(T_K: float, mu_solvent_cP: float, M_solvent: float, phi_solvent: float, V_solute_cm3_mol: float) -> float:
    """Dilute liquid diffusivity (m²/s): 7.4e-8 (φ M)^0.5 T/(μ V_A^0.6) cm²/s. φ = solvent association factor,
    V_A = solute molar volume at its normal boiling point."""
    return 7.4e-8 * math.sqrt(phi_solvent * M_solvent) * T_K / (mu_solvent_cP * V_solute_cm3_mol ** 0.6) * 1e-4


def fuller(T_K: float, P_Pa: float, M_A: float, M_B: float, sum_v_A: float, sum_v_B: float) -> float:
    """Binary gas diffusivity (m²/s), Fuller-Schettler-Giddings: 1e-3 T^1.75 (1/M_A + 1/M_B)^0.5 / (P_atm (Σv_A^⅓ + Σv_B^⅓)²) cm²/s."""
    P_atm = P_Pa / 101325.0
    return 1e-3 * T_K ** 1.75 * math.sqrt(1.0 / M_A + 1.0 / M_B) / (P_atm * (sum_v_A ** (1 / 3) + sum_v_B ** (1 / 3)) ** 2) * 1e-4
