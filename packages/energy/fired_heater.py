"""Radiant and convection sections of fired heaters and boilers (bloque M18, part B).

Radiant section: one well-stirred gas zone (the Lobo-Evans / Hottel one-gas-zone idea). The flue gas is grey at the gas
temperature, the tubes are a grey sink, the refractory is radiatively adiabatic and the enclosure is «speckled» (every surface
sees sink and refractory in proportion to their areas). The radiation network gives the total exchange area

    1/GS = (1 - e_t)/(e_t A1) + 1 / [ e_g A1 + 1 / ( 1/(e_g A_R) + A_T/(A1 A_R (1 - e_g)) ) ]      A1 = alpha*Acp (+ shield)

and F = GS/A1 is the overall exchange factor: Q_rad = sigma F A1 (Tg^4 - Tw^4). The firebox heat balance fixes Tg, which is
also the bridgewall temperature (complete mixing). Flue-gas enthalpy comes from `packages.energy.combustion` (NASA).

Sourced numbers (nothing typed from memory):
  * Gas emissivity: weighted sum of grey gases of Smith, Shen and Friedman (J. Heat Transfer 104 (1982) 602-608), 3 grey
    gases + clear, coefficients as reproduced in Table 7 of Marzouk and Huckaby, «New Weighted Sum of Gray Gases (WSGG) Models
    for Radiation Calculation in Carbon Capture Simulations», 7th U.S. National Combustion Meeting 2011 (arXiv:2411.18467),
    read 2026-11-14; mixtures by the piecewise-linear rule of its Appendix B. Valid at 1 atm total, 600-2400 K,
    0.001-10 atm m.
  * Mean beam length 3.6 V/A: the same paper («The mean path length, L, was approximated by 3.6 volume/area»).
  * Tube-row effectiveness alpha: exact crossed-strings view factor of a plane to one row of tubes, plus the re-radiation of
    the refractory behind it, alpha = F (2 - F); a test integrates the view factor numerically.
  * Convection across the bank: Žukauskas, «Heat Transfer from Tubes in Crossflow», Advances in Heat Transfer 8 (1972)
    93-160, eqs. (39)-(44), read 2026-11-14 in the scan of the volume at the Internet Archive (identifier
    AdvancesInHeatTransfer) and coded here in `zukauskas_bank_nusselt` with the arrangement given explicitly. The row-count
    factor is the table digitised in `ht.conv_tube_bank.Zukauskas_tube_row_correction`. `ht`'s own
    `Nu_Zukauskas_Bejan` (ht 1.2.0) is NOT used: it guesses the arrangement from the pitch ratio (an in-line bank with
    St != Sl is computed as staggered), has the exponent 0.05 instead of 0.5 for in-line banks at Re 100-1000 and lacks the
    St/Sl > 2 branch. Gas viscosity, conductivity and cp from Cantera (GRI-Mech 3.0 transport data, mixture averaged).

Validation (bloque M18b, 2026-11-14; numbers and sources in tests/test_fired_heater_validation.py):
  * Gas emissivity: against Leckner's 1972 correlation (independent), 70 points 800-2000 K, 0.03-3 atm m: -12 % to +18 %;
    at the two points of Kern's examples both agree (0.398 vs 0.39-0.40; 0.115 vs 0.12) and Kern's 1950 charts are the
    outlier at long path (0.496). The WSGG of Smith et al. stays as the only model.
  * Radiant section: Kern (1950) examples 19.1 (Lobo-Evans) and 19.2 (Wilson-Lobo-Hottel) reproduced: radiant duty
    -3.2 % and -1.7 %, bridgewall +33 K and +11 K with this module's own emissivity; +0.2 % and +2 K with Kern's emissivity.
  * Convection: the bank coefficient reproduces Incropera example 7.7 (staggered, +0.7 % on the FOSSEE transcription) and
    Cengel example 7-7 (in-line, +0.2 %); the gas radiation between tubes is 10 % above Kern's example 19.4 with his beam
    length; the convection section of Nelson's (1941) example 68 comes out with +5 % in the overall coefficient and +2 % in
    the heat given up by the gas.
  * Radiant section, bloque L3 (2026-11-15; tests/test_fired_heater_cases.py): the box heater of the Esteem Projects fired
    heater tutorial (published gas, 843 lb/h, 9.5e6 Btu/h at ~1600 °F) gives -1.4 % duty and +10 to +13 K; the vertical
    cylindrical heater of the Heat Transfer Consult API 560 data sheets gives a bridgewall 5 K under its Lobo-Evans ~867 °C.
  * Studded tubes (bloque L3): the stud correlation printed by the same Esteem tutorial, coded as printed. No worked example
    with printed numbers was found: NOT validated, only its bare-tube limit and identities are checked.

Every operating input (firing, geometry, tube emissivity, wall temperature or inside coefficient, convective coefficient in
the firebox, setting loss, allowable flux) is the caller's; nothing typical is assumed."""
from __future__ import annotations

import math
from functools import lru_cache

import cantera as ct
from scipy.optimize import brentq

from packages.energy import combustion as CB

SIGMA = ct.stefan_boltzmann            # W/(m2 K4)
ATM = ct.one_atm
WSGG_SOURCE = ("Smith, Shen y Friedman, J. Heat Transfer 104 (1982) 602-608; coeficientes leídos de la Tabla 7 de Marzouk y "
               "Huckaby 2011 (arXiv:2411.18467)")
# node -> (k_i [1/(atm m)], b_ij: a_i = sum_j b_ij T^(j-1))
SMITH_1982 = {
    "CO2_limit": ((0.3966, 15.64, 394.3), ((0.4334e-1, 2.620e-4, -1.560e-7, 2.565e-11),
                                           (-0.4814e-1, 2.822e-4, -1.794e-7, 3.274e-11),
                                           (0.5492e-1, 0.1087e-4, -0.3500e-7, 0.9123e-11))),
    "ratio_1": ((0.4303, 7.055, 178.1), ((5.150e-1, -2.303e-4, 0.9779e-7, -1.494e-11),
                                         (0.7749e-1, 3.399e-4, -2.297e-7, 3.770e-11),
                                         (1.907e-1, -1.824e-4, 0.5608e-7, -0.5122e-11))),
    "ratio_2": ((0.4201, 6.516, 131.9), ((6.508e-1, -5.551e-4, 3.029e-7, -5.353e-11),
                                         (-0.2504e-1, 6.112e-4, -3.882e-7, 6.528e-11),
                                         (2.718e-1, -3.118e-4, 1.221e-7, -1.612e-11))),
    "H2O_limit": ((0.4098, 6.325, 120.5), ((5.977e-1, -5.119e-4, 3.042e-7, -5.564e-11),
                                           (0.5677e-1, 3.333e-4, -1.967e-7, 2.718e-11),
                                           (1.800e-1, -2.334e-4, 1.008e-7, -1.454e-11))),
    "H2O_1atm": ((0.4496, 7.113, 119.7), ((6.324e-1, -8.358e-4, 6.135e-7, -13.03e-11),
                                          (-0.2016e-1, 7.145e-4, -5.212e-7, 9.868e-11),
                                          (3.500e-1, -5.040e-4, 2.425e-7, -3.888e-11))),
}


def _need(d: dict, keys, what: str) -> None:
    missing = [k for k in keys if d.get(k) is None]
    if missing:
        raise ValueError(f"{what}: faltan {missing}; son datos del equipo o de operación y no se suponen")


def gas_emissivity(T_K: float, p_H2O_Pa: float, p_CO2_Pa: float, path_length_m: float, total_pressure_Pa: float = ATM) -> dict:
    """Total emissivity of an H2O-CO2 mixture in a transparent gas (WSGG, Smith et al. 1982)."""
    pw, pc = p_H2O_Pa / ATM, p_CO2_Pa / ATM
    if pw < 0 or pc < 0 or pw + pc <= 0 or path_length_m <= 0 or T_K <= 0:
        raise ValueError("gas_emissivity: da presiones parciales no negativas (alguna positiva), longitud y T positivas")
    rr = pw / (pw + pc)
    top = "H2O_limit" if pw <= 0.5 else "H2O_1atm"
    nodes = ((0.0, "CO2_limit"), (0.5, "ratio_1"), (2.0 / 3.0, "ratio_2"), (1.0, top))
    for (r0, n0), (r1, n1) in zip(nodes, nodes[1:]):
        if rr <= r1 + 1e-12:
            break
    u = (rr - r0) / (r1 - r0)
    (k0, b0), (k1, b1) = SMITH_1982[n0], SMITH_1982[n1]
    pL = (pw + pc) * path_length_m
    eps, weights = 0.0, []
    for i in range(3):
        k = k0[i] + u * (k1[i] - k0[i])
        a = sum((b0[i][j] + u * (b1[i][j] - b0[i][j])) * T_K ** j for j in range(4))
        weights.append(a)
        eps += a * (1.0 - math.exp(-k * pL))
    out_of_range = []
    if not 600.0 <= T_K <= 2400.0:
        out_of_range.append(f"T = {T_K:.0f} K fuera de 600-2400 K")
    if not 0.001 <= pL <= 10.0:
        out_of_range.append(f"pL = {pL:.4g} atm·m fuera de 0.001-10")
    if abs(total_pressure_Pa / ATM - 1.0) > 0.1:
        out_of_range.append(f"presión total {total_pressure_Pa/ATM:.2f} atm; el ajuste es para 1 atm")
    if min(weights) < 0 or not 0 < eps < 1:
        out_of_range.append("pesos o emisividad no físicos")
    return {"emissivity": eps, "pL_atm_m": pL, "H2O_fraction_of_absorbers": rr, "interpolated_between": [n0, n1],
            "in_fitted_range": not out_of_range, "out_of_range": out_of_range, "source": WSGG_SOURCE}


def tube_row_alpha(pitch_m: float, diameter_m: float) -> dict:
    """Fraction of the radiation crossing the cold plane that one row of tubes in front of a refractory wall absorbs."""
    if not 0 < diameter_m <= pitch_m:
        raise ValueError("tube_row_alpha: el paso debe ser mayor o igual que el diámetro exterior")
    x = diameter_m / pitch_m
    direct = 1.0 - math.sqrt(1.0 - x * x) + x * math.atan(math.sqrt(1.0 / (x * x) - 1.0)) if x < 1 else 1.0
    return {"direct_view_factor": direct, "alpha": direct * (2.0 - direct), "pitch_to_diameter": 1.0 / x}


def exchange_factor(gas_emissivity_value: float, sink_area_m2: float, envelope_area_m2: float, tube_emissivity: float) -> float:
    """Overall exchange factor F on the sink area (see the module docstring)."""
    eg, A1, AT, et = gas_emissivity_value, sink_area_m2, envelope_area_m2, tube_emissivity
    if not (0 < eg < 1 and 0 < et <= 1 and 0 < A1 <= AT * (1 + 1e-9)):
        raise ValueError("exchange_factor: emisividades en (0, 1] y área de sumidero no mayor que la envolvente")
    AR = max(AT - A1, 0.0)
    via_refractory = 0.0 if AR <= 0 else 1.0 / (1.0 / (eg * AR) + AT / (A1 * AR * (1.0 - eg)))
    return 1.0 / ((1.0 - et) / (et * A1) + 1.0 / (eg * A1 + via_refractory)) / A1


def _wall_resistance(p: dict, Do: float) -> float:
    """(Tw_outside - T_bulk) per unit outside flux, m2 K/W: inside film + inside fouling + tube wall."""
    Di = p["tube_inside_diameter_m"]
    if not 0 < Di < Do:
        raise ValueError("tube_inside_diameter_m debe ser menor que el diámetro exterior")
    return Do / (Di * p["inside_coefficient_W_m2K"]) + p["inside_fouling_m2K_W"] * Do / Di + \
        Do * math.log(Do / Di) / (2.0 * p["wall_conductivity_W_mK"])


def gas_radiation_coefficient(gas_T_K: float, wall_T_K: float, gas_emissivity_value: float, tube_emissivity: float) -> float:
    """Radiation coefficient of a grey gas to the tube surface that encloses it, W/(m2 K)."""
    Fg = 1.0 / (1.0 / gas_emissivity_value + 1.0 / tube_emissivity - 1.0)
    return SIGMA * Fg * (gas_T_K ** 4 - wall_T_K ** 4) / (gas_T_K - wall_T_K)


def bank_beam_length(transverse_pitch_m: float, longitudinal_pitch_m: float, diameter_m: float) -> float:
    """3.6 V/A of the gas space around one tube of a bank, m."""
    return 3.6 * (transverse_pitch_m * longitudinal_pitch_m - math.pi * diameter_m ** 2 / 4) / (math.pi * diameter_m)


RADIANT_VALIDATION = (
    "Validado contra dos ejemplos resueltos de Kern, Process Heat Transfer (1950), cap. 19, mismo hogar: 19.1 (Lobo y Evans, "
    "fuel oil) carga radiante -3.2 % y pared divisoria +33 K; 19.2 (Wilson, Lobo y Hottel, gas) -1.7 % y +11 K. Con la "
    "emisividad de las cartas de Kern en vez de la propia: +0.2 % y +2 K. La emisividad WSGG de Smith 1982 coincide con la "
    "correlación independiente de Leckner 1972 (-12 % a +18 % en 70 puntos, 800-2000 K y 0.03-3 atm·m; 1-2 % en el punto del "
    "ejemplo); las cartas de Kern quedan 23-27 % arriba de ambas a trayectoria larga. Banda a esperar: carga radiante ±5 %, "
    "pared divisoria ±35 K, en hogares de caja de una fila con llama no luminosa. El análisis del combustible de los "
    "ejemplos de Kern no está publicado y se dedujo de sus propios datos. Dos hogares más (bloque L3, 2026-11-15): hogar de "
    "caja del tutorial de Esteem Projects (gas de composición publicada, 843 lb/h, 9.5e6 Btu/h a ~1600 °F): carga -1.4 % y "
    "pared divisoria +10 a +13 K; calentador cilíndrico vertical con hoja de datos API 560 completa de Heat Transfer Consult "
    "(metano, 26.26 MW liberados): pared divisoria -5 K frente a su cálculo Lobo-Evans (~867 °C), con el coeficiente "
    "convectivo del hogar de la regla de ese tutorial (3 Btu/h ft² °F) y la temperatura media de pared estimada de su hoja.")
CONVECTION_VALIDATION = (
    "Validado por partes. Coeficiente de convección del banco: Incropera ej. 7.7 (tresbolillo, transcripción FOSSEE) +0.7 % con "
    "las propiedades del libro y +1.9 % con las de Cantera; Çengel ej. 7-7 (en línea) +0.2 %; Grimison y ESDU 73031 a "
    "-7 %...+10 %. Radiación del gas entre tubos: Kern ej. 19.4, +10 % con su longitud de haz y +32 % con 3.6 V/A (paso 1.7 D). Sección completa: "
    "Nelson, Petroleum Refinery Engineering (1941), ej. 68: coeficiente global +5 %, carga +2 % sobre el calor que cede el gas "
    "(+12 % sobre lo que absorbe el aceite: el ejemplo descuenta por paredes del banco 2 % del calor liberado, que este modelo "
    "no tiene) y chimenea +48 K (su carta de entalpías da 2.8 puntos menos a 1700 °F). El paso, el arreglo y la emisividad de "
    "tubo de ese ejemplo no están publicados y se supusieron. Banda a esperar: coeficiente de convección ±10 %, carga de la "
    "sección ±12 %, chimenea ±50 K. Solo tubos lisos.")
ZUKAUSKAS_SOURCE = ("A. Žukauskas, «Heat Transfer from Tubes in Crossflow», Advances in Heat Transfer 8 (1972) 93-160, ecs. "
                    "(39)-(44); factor por número de filas de ht.conv_tube_bank.Zukauskas_tube_row_correction")


def bank_free_width(duct_width_m: float, tubes_per_row: int, diameter_m: float, transverse_pitch_m: float,
                    longitudinal_pitch_m: float, arrangement: str) -> float:
    """Width of the minimum free cross-section of a bank, m (the maximum velocity is the flow divided by it and the length).

    In-line: the transverse gaps. Staggered: the smaller of the transverse gaps and the two diagonal gaps per tube."""
    gap = duct_width_m - tubes_per_row * diameter_m
    if arrangement == "staggered":
        diagonal = math.sqrt(longitudinal_pitch_m ** 2 + (transverse_pitch_m / 2) ** 2)
        gap = min(gap, tubes_per_row * 2 * (diagonal - diameter_m))
    return gap


def zukauskas_bank_nusselt(Re: float, Pr: float, Pr_wall: float, n_rows: int, transverse_pitch_m: float,
                           longitudinal_pitch_m: float, arrangement: str) -> dict:
    """Mean Nusselt number of a bank of bare tubes in cross flow (Žukauskas 1972).

    Re on the tube diameter and the velocity in the minimum free section, properties at the mean bulk temperature, Pr_wall
    at the wall temperature. Inner rows: eq. (39) in-line and eqs. (40)-(41) staggered for 1e3 <= Re < 2e5, eqs. (42)-(43)
    above; the result is multiplied by the row-count factor (banks of fewer than 20 rows). Below Re = 1e3 the constants
    are those listed in the documentation of `ht` (Bejan, Convection Heat Transfer, 4th ed.), not read in the 1972 text."""
    if arrangement not in ("staggered", "inline"):
        raise ValueError("arrangement: 'staggered' o 'inline'")
    if Re <= 0 or Pr <= 0 or Pr_wall <= 0 or n_rows < 1:
        raise ValueError("zukauskas_bank_nusselt: Re, Pr, Pr_wall positivos y al menos una fila")
    from ht.conv_tube_bank import Zukauskas_tube_row_correction
    staggered, ratio = arrangement == "staggered", transverse_pitch_m / longitudinal_pitch_m
    notes = []
    if Re < 1e3:
        if staggered:
            c, m, eq = (1.04, 0.4, "Bejan (vía ht), Re < 500") if Re < 500 else (0.71, 0.5, "Bejan (vía ht), 500 <= Re < 1e3")
        else:
            c, m, eq = (0.9, 0.4, "Bejan (vía ht), Re < 100") if Re < 100 else (0.52, 0.5, "Bejan (vía ht), 100 <= Re < 1e3")
        notes.append("Reynolds < 1e3: constantes de la lista de Bejan (documentación de ht), no leídas en Žukauskas 1972 y "
                     "fuera de lo validado")
    elif Re < 2e5:
        if staggered:
            c, m, eq = (0.35 * ratio ** 0.2, 0.60, "ec. (40)") if ratio < 2 else (0.40, 0.60, "ec. (41)")
        else:
            c, m, eq = 0.27, 0.63, "ec. (39)"
    else:
        c, m, eq = (0.022, 0.84, "ec. (43)") if staggered else (0.021, 0.84, "ec. (42)")
        if Re > 2e6:
            notes.append("Reynolds > 2e6: fuera de los datos de Žukauskas (20 a 2e6)")
    if not staggered and ratio < 0.7:
        notes.append("banco en línea con paso transversal / longitudinal < 0.7: Žukauskas avisa que la transferencia medida es "
                     "mucho menor que la de la ec. (39)")
    rows = Zukauskas_tube_row_correction(int(n_rows), staggered=staggered, Re=Re)
    inner = c * Re ** m * Pr ** 0.36 * (Pr / Pr_wall) ** 0.25
    return {"Nu": inner * rows, "Nu_inner_rows": inner, "row_factor": rows, "C": c, "m": m, "equation": eq,
            "in_source_range": not notes, "notes": notes}


ESCOA_SOURCE = ("ESCOA revisada (1993; Fintube Technologies, Tulsa) tal como la imprime V. Ganapathy, Steam Generators and Waste "
                "Heat Boilers, CRC Press 2015, apéndice E: Tabla E.2 (C1, C3, C5) y ecs. (E.1)-(E.15), (E.22)-(E.23)")
FIN_TYPES = ("solid_fin", "serrated_fin")
FIN_KEYS = ("fin_height_m", "fin_thickness_m", "fins_per_m", "fin_conductivity_W_mK")
FINNED_VALIDATION = (
    "Tubos aletados (ESCOA revisada, Ganapathy 2015 ap. E). EN LÍNEA, contra sus ejemplos resueltos: E.10 (aleta segmentada) "
    "h_c +0.5 %, eficiencia y efectividad de aleta +0.4 %; E.1 (aleta sólida) h_c -0.01 % con sus propiedades, eficiencia de "
    "aleta -3 % (aquí Bessel exacto; su tabla de Bessel está mal leída, su fórmula simplificada da +0.4 %) y U -1.6 %; E.1 como "
    "sección completa con propiedades de Cantera y radiación WSGG: carga +2.9 %, U +3.6 %, gas de salida -2 K. EN TRESBOLILLO "
    "no hay ejemplo resuelto con las constantes revisadas: contra el de Esteem Projects (aleta segmentada, versión anterior de "
    "ESCOA) h_c -7 %. Límite de tubo liso (aleta -> 0) frente a Žukauskas: tresbolillo -9 % a +18 %; en línea -25 % a -58 % "
    "(ESCOA no se extrapola a tubo liso: con aletas muy bajas usa tube_type='bare'). Los números de Ganapathy se transcribieron "
    "de una copia cuya autorización no se pudo verificar y no se halló una fuente abierta legítima de esos ejemplos: pendiente "
    "cotejarlos con un ejemplar con licencia. Banda a esperar: h_c y U ±5 % en línea, ±10 % en tresbolillo; carga de la "
    "sección ±5 %.")
STUD_KEYS = ("stud_length_m", "stud_diameter_m", "studs_per_plane", "stud_planes_per_m", "stud_conductivity_W_mK")
STUD_SOURCE = ("Esteem Projects, «Free Fired Heater Design Tutorial - Convection Section Design», sección «Convection Transfer, "
               "Stud Tubes» (fórmulas y su guion de cálculo, leídos 2026-11-15: "
               "https://www.esteemprojects.com/library/Fired-Heater-Design/page4_3.htm); la fuente no cita de dónde viene la "
               "correlación")
STUDDED_VALIDATION = (
    "NO validado contra un ejemplo resuelto: no se encontró ninguno abierto con números impresos para tubos con pernos. Lo "
    "comprobado: las fórmulas del tutorial de Esteem Projects reproducidas tal cual (unidades inglesas en la prueba, SI aquí), "
    "su límite sin pernos (h_e -> h_t, coeficiente de tubo liso de la misma fuente, que según ella ya incluye la radiación del "
    "gas) frente al tubo liso de este módulo: h_t = convección de Žukauskas -1 %, pero 31 % bajo convección + radiación WSGG "
    "(carga -16 % en un banco de prueba con gas a ~1540 °F), así que la radiación «incluida» no aparece y la carga con pernos "
    "puede quedar subestimada; además cierre del balance y monotonía con el número, largo y conductividad de los pernos. Trata "
    "la carga con incertidumbre no medida; para diseño, cotéjala con el fabricante.")
STUD_BTU_H_FT2_F = 5.678263            # W/(m2 K) per Btu/(h ft2 °F)
STUD_LB_H_FT2 = 0.45359237 / 3600.0 / 0.09290304   # kg/(m2 s) per lb/(h ft2)


def finned_tube_geometry(diameter_m: float, fin_type: str, fin_height_m: float, fin_thickness_m: float, fins_per_m: float,
                         segment_width_m: float | None = None) -> dict:
    """Areas of one metre of finned tube as Ganapathy (2015) app. E writes them.

    Obstruction width d + 2 n b h (eq. E.4), fin area (E.7 solid, E.11 serrated with segment width ws), total outside area
    A_t = A_f + pi d (1 - n b) (E.8), clear gap between fins s = 1/n - b."""
    d, h, b, n = diameter_m, fin_height_m, fin_thickness_m, fins_per_m
    if fin_type not in FIN_TYPES:
        raise ValueError("bank.tube_type: 'bare', 'solid_fin' o 'serrated_fin'")
    if min(d, h, b, n) <= 0 or n * b >= 1.0:
        raise ValueError("aletas: diámetro, altura, espesor y densidad positivos, y espesor × densidad < 1")
    if fin_type == "solid_fin":
        A_f = math.pi * n * (2 * d * h + 2 * h ** 2 + b * d + 2 * b * h)
    else:
        if not segment_width_m or segment_width_m <= 0:
            raise ValueError("aletas segmentadas: falta bank.fin_segment_width_m (ancho del segmento, m; entra en el área de "
                             "aleta y en su eficiencia)")
        ws = segment_width_m
        A_f = math.pi * d * n * (2 * h * (ws + b) + b * ws) / ws
    A_t = A_f + math.pi * d * (1 - n * b)
    return {"obstruction_m": d + 2 * n * b * h, "fin_area_m2_per_m": A_f, "total_area_m2_per_m": A_t,
            "fin_gap_m": 1.0 / n - b, "area_ratio_to_bare": A_t / (math.pi * d)}


def escoa_colburn(Re: float, n_rows: int, transverse_pitch_m: float, longitudinal_pitch_m: float, diameter_m: float,
                  fin_height_m: float, fin_gap_m: float, fin_type: str, arrangement: str) -> dict:
    """Revised ESCOA factors (Ganapathy 2015, Table E.2) and the geometric part of the Colburn factor.

    h_c = C1 C3 C5 ((d + 2h)/d)^0.5 (T_gas/T_fin)^x G cp Pr^-0.67, x = 0.5 in-line and 0.25 staggered (eqs. E.1-E.2), Re on
    the tube diameter and the mass velocity in the free area between obstructions. C1 in-line is coded as in Ganapathy's
    worked examples E.1 and E.10: 0.053 (1.45 - 2.9 (S_L/d)^-2.3) Re^-0.21."""
    if arrangement not in ("staggered", "inline") or fin_type not in FIN_TYPES:
        raise ValueError("escoa_colburn: arrangement 'staggered'|'inline' y fin_type 'solid_fin'|'serrated_fin'")
    d, h, s, N, St, Sl = diameter_m, fin_height_m, fin_gap_m, int(n_rows), transverse_pitch_m, longitudinal_pitch_m
    if Re <= 0 or s <= 0 or N < 1:
        raise ValueError("escoa_colburn: Re y separación entre aletas positivos, al menos una fila")
    if arrangement == "inline":
        geometric = 1.45 - 2.9 * (Sl / d) ** -2.3
        if geometric <= 0:
            raise ValueError("ESCOA en línea: paso longitudinal / diámetro demasiado chico (1.45 - 2.9 (S_L/d)^-2.3 <= 0)")
        C1 = 0.053 * geometric * Re ** -0.21
        C5 = 1.1 - (0.75 - 1.5 * math.exp(-0.7 * N)) * math.exp(-2.0 * Sl / St)
        C3 = 0.20 + 0.65 * math.exp(-0.25 * h / s) if fin_type == "solid_fin" else 0.25 + 0.6 * math.exp(-0.26 * h / s)
        x = 0.5
    else:
        C1 = 0.091 * Re ** -0.25
        C5 = 0.7 + (0.7 - 0.8 * math.exp(-0.15 * N ** 2)) * math.exp(-1.0 * Sl / St)
        C3 = 0.35 + 0.65 * math.exp(-0.25 * h / s) if fin_type == "solid_fin" else 0.35 + 0.65 * math.exp(-0.17 * h / s)
        x = 0.25
    return {"C1": C1, "C3": C3, "C5": C5, "temperature_exponent": x, "j_geometric": C1 * C3 * C5 * ((d + 2 * h) / d) ** 0.5}


def fin_efficiency(fin_type: str, outside_coefficient_W_m2K: float, fin_conductivity_W_mK: float, diameter_m: float,
                   fin_height_m: float, fin_thickness_m: float, segment_width_m: float | None = None) -> dict:
    """Fin efficiency E and fin-tip temperature ratio (T_gas - T_tip)/(T_gas - T_base), Ganapathy (2015) app. E.

    Solid (annular) fin, adiabatic tip: Bessel solution (E.9), m = (2 h_o / (k b))^0.5, tip ratio (E.22). Serrated fin:
    straight fin E = tanh(mh)/(mh) with m = (2 h_o (b + ws) / (k b ws))^0.5 (E.13-E.14); tip ratio 1/cosh(mh) (Ganapathy prints
    cosh(mb); the straight-fin solution is cosh(mh), which is what is coded)."""
    from scipy.special import i0, i1, k0, k1
    ho, k, d, h, b = outside_coefficient_W_m2K, fin_conductivity_W_mK, diameter_m, fin_height_m, fin_thickness_m
    if ho <= 0 or k <= 0:
        raise ValueError("fin_efficiency: coeficiente exterior y conductividad de la aleta positivos")
    if fin_type == "solid_fin":
        m = math.sqrt(2 * ho / (k * b))
        ro, re = d / 2, d / 2 + h
        den = i0(m * ro) * k1(m * re) + i1(m * re) * k0(m * ro)
        E = 2 * ro / (m * (re ** 2 - ro ** 2)) * (i1(m * re) * k1(m * ro) - k1(m * re) * i1(m * ro)) / den
        tip = (k1(m * re) * i0(m * re) + i1(m * re) * k0(m * re)) / den
    else:
        ws = segment_width_m
        m = math.sqrt(2 * ho * (b + ws) / (k * b * ws))
        E = math.tanh(m * h) / (m * h)
        tip = 1.0 / math.cosh(m * h)
    return {"E": float(E), "m_1_m": m, "mh": m * h, "tip_ratio": float(tip)}


def finned_bank_beam_length(transverse_pitch_m: float, longitudinal_pitch_m: float, diameter_m: float,
                            fin_height_m: float) -> float:
    """Ganapathy (2015) eq. E.15: L = (d + 2h) (S_T S_L / (d + 2h)^2 - 0.85), m."""
    D = diameter_m + 2 * fin_height_m
    return D * (transverse_pitch_m * longitudinal_pitch_m / D ** 2 - 0.85)


def studded_tube_geometry(diameter_m: float, stud_length_m: float, stud_diameter_m: float, studs_per_plane: float,
                          stud_planes_per_m: float) -> dict:
    """Areas of one metre of studded tube, as the Esteem Projects stud script computes them [STUD_SOURCE].

    Stud area A_fo = (pi D_s L_s + pi D_s^2/4) n_s n_p (side + tip), exposed tube area A_po = pi d - (pi D_s^2/4) n_s n_p,
    obstruction width for the free flow area d + L_s D_s n_p (the script's od + lf tf pf / 12)."""
    d, Ls, Ds, ns, npl = diameter_m, stud_length_m, stud_diameter_m, studs_per_plane, stud_planes_per_m
    if min(d, Ls, Ds, ns, npl) <= 0:
        raise ValueError("pernos: diámetro de tubo, largo y diámetro del perno, pernos por plano y planos por metro positivos")
    foot = math.pi * Ds ** 2 / 4 * ns * npl
    if foot >= math.pi * d:
        raise ValueError("pernos: las bases de los pernos cubren todo el tubo (revisa pernos por plano y planos por metro)")
    if ns * Ds >= math.pi * d or npl * Ds >= 1.0:
        raise ValueError("pernos: no caben (pernos por plano × diámetro del perno >= perímetro, o planos × diámetro >= 1 m)")
    A_fo = (math.pi * Ds * Ls + math.pi * Ds ** 2 / 4) * ns * npl
    A_po = math.pi * d - foot
    return {"obstruction_m": d + Ls * Ds * npl, "stud_area_m2_per_m": A_fo, "tube_area_m2_per_m": A_po,
            "total_area_m2_per_m": A_fo + A_po, "area_ratio_to_bare": (A_fo + A_po) / (math.pi * d)}


def stud_coefficients(mass_velocity_kg_m2s: float, gas_T_K: float, diameter_m: float, stud_length_m: float,
                      stud_diameter_m: float, stud_conductivity_W_mK: float, outside_fouling_m2K_W: float, geometry: dict,
                      arrangement: str, longitudinal_pitch_m: float) -> dict:
    """Effective outside coefficient of a studded tube, W/(m2 K), with the Esteem Projects stud correlation [STUD_SOURCE].

    In its units (G_n lb/(h ft2) in the free area, T_b mean gas °F, d inches): base tube h_t = 0.717 d^-0.333 (G_n/1000)^0.67
    (T_b + 460)^0.3, stud h_s = 0.936 (G_n/1000)^0.67 (T_b + 460)^0.3, both Btu/(h ft2 °F) and with the gas radiation «built
    into the equations»; h_so = 1/(1/h_s + R_fo); stud efficiency E = 1/((e^X + e^-X)/1.950), X = L_s (2 h_so/(k_s D_s))^0.5;
    h_e = (h_so E A_fo + h_t A_po)/A_o, times (d/P_l)^0.333 for an in-line bank. Coded as printed: the outside fouling is
    applied to the studs only. E is a fit of the pin fin with adiabatic tip, tanh(mL)/(mL) with m = (4 h/(k D_s))^0.5 (mL =
    X sqrt 2): within ±2.4 % for X <= 1.2, -9.5 % at X = 1.5 and -26 % at X = 2 (measured in tests/test_fired_heater_cases.py)."""
    if arrangement not in ("staggered", "inline"):
        raise ValueError("arrangement: 'staggered' o 'inline'")
    if mass_velocity_kg_m2s <= 0 or gas_T_K <= 0 or stud_conductivity_W_mK is None or stud_conductivity_W_mK <= 0:
        raise ValueError("stud_coefficients: velocidad másica, temperatura y conductividad del perno positivas")
    Gn = mass_velocity_kg_m2s / STUD_LB_H_FT2
    TR = (gas_T_K - 273.15) * 1.8 + 32.0 + 460.0
    common = (Gn / 1000.0) ** 0.67 * TR ** 0.3
    h_t = 0.717 / (diameter_m / 0.0254) ** 0.333 * common * STUD_BTU_H_FT2_F
    h_s = 0.936 * common * STUD_BTU_H_FT2_F
    h_so = 1.0 / (1.0 / h_s + float(outside_fouling_m2K_W))
    X = stud_length_m * math.sqrt(2.0 * h_so / (stud_conductivity_W_mK * stud_diameter_m))
    E = 1.950 / (math.exp(X) + math.exp(-X))
    g = geometry
    h_e = (h_so * E * g["stud_area_m2_per_m"] + h_t * g["tube_area_m2_per_m"]) / g["total_area_m2_per_m"]
    inline_factor = (diameter_m / longitudinal_pitch_m) ** 0.333 if arrangement == "inline" else 1.0
    return {"h_t": h_t, "h_s": h_s, "h_so": h_so, "X": X, "E": E, "h_e": h_e * inline_factor, "inline_factor": inline_factor,
            "Gn_lb_h_ft2": Gn, "Tb_R": TR}


PROCESS_KEYS = ("bulk_T_K", "inside_coefficient_W_m2K", "tube_inside_diameter_m", "wall_conductivity_W_mK",
                "inside_fouling_m2K_W")


def radiant_section(fuel: dict, air: dict, pressure_Pa: float, fuel_flow_kg_s: float, geometry: dict, tube_emissivity: float,
                    convective_coefficient_W_m2K: float, setting_loss_fraction: float, setting_loss_basis: str,
                    excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                    flue_CO_dry_mole_fraction: float | None = None, tube_wall_T_K: float | None = None,
                    process: dict | None = None, gas_emissivity_override: float | None = None,
                    exchange_factor_override: float | None = None) -> dict:
    """Rating of the radiant section: bridgewall temperature, duty absorbed and average flux for a firing rate and a geometry."""
    c = CB.solve_combustion(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction)
    if fuel_flow_kg_s is None or fuel_flow_kg_s <= 0:
        raise ValueError("fuel_flow_kg_s: da el combustible quemado (combustion_efficiency lo calcula para una carga)")
    if setting_loss_fraction is None or setting_loss_basis not in ("HHV", "LHV") or not 0 <= setting_loss_fraction < 0.5:
        raise ValueError("da setting_loss_fraction (pérdida por paredes del hogar, 0 si es nula) y setting_loss_basis")
    if tube_emissivity is None or convective_coefficient_W_m2K is None or convective_coefficient_W_m2K < 0:
        raise ValueError("da tube_emissivity y convective_coefficient_W_m2K (0 si no hay convección en el hogar)")
    g = geometry or {}
    _need(g, ("tube_outside_diameter_m", "tube_pitch_m", "tube_effective_length_m", "n_tubes", "envelope_area_m2"), "geometry")
    if g.get("mean_beam_length_m") is None and g.get("volume_m3") is None and gas_emissivity_override is None \
            and exchange_factor_override is None:
        raise ValueError("geometry: da volume_m3 (la longitud media de haz se toma como 3.6 V/A) o mean_beam_length_m")
    if (tube_wall_T_K is None) == (process is None):
        raise ValueError("da tube_wall_T_K (temperatura media de pared de tubo) o process {bulk_T_K, inside_coefficient_W_m2K, "
                         "tube_inside_diameter_m, wall_conductivity_W_mK, inside_fouling_m2K_W}, uno solo")
    Do, pitch, L, n = (g["tube_outside_diameter_m"], g["tube_pitch_m"], g["tube_effective_length_m"], g["n_tubes"])
    A_tubes, A_cp = n * math.pi * Do * L, n * pitch * L
    row = tube_row_alpha(pitch, Do)
    alpha = g["alpha"] if g.get("alpha") is not None else row["alpha"]
    A_shield = float(g.get("shield_cold_plane_area_m2") or 0.0)
    A1, AT = alpha * A_cp + A_shield, g["envelope_area_m2"]
    if A1 > AT:
        raise ValueError("el plano frío equivalente supera el área de la envolvente: revisa envelope_area_m2 (área interior "
                         "total del hogar, incluida la que cubren los tubos)")
    beam = g.get("mean_beam_length_m") or (3.6 * g["volume_m3"] / AT if g.get("volume_m3") else None)
    wet = c.flue_mol
    p_w, p_c = c.flue.get("H2O", 0.0) / wet * pressure_Pa, c.flue.get("CO2", 0.0) / wet * pressure_Pa
    m = float(fuel_flow_kg_s)
    setting = m * setting_loss_fraction * (c.HHV if setting_loss_basis == "HHV" else c.LHV)
    released = m * (c.LHV + c.credits - c.co_loss)
    R_wall = None
    if process is not None:
        _need(process, PROCESS_KEYS, "process")
        R_wall = _wall_resistance(process, Do)

    def state(Tg: float) -> dict:
        em = {"emissivity": gas_emissivity_override} if gas_emissivity_override is not None else \
            (gas_emissivity(Tg, p_w, p_c, beam, pressure_Pa) if exchange_factor_override is None else {"emissivity": None})
        F = exchange_factor_override if exchange_factor_override is not None else \
            exchange_factor(em["emissivity"], A1, AT, tube_emissivity)

        def duty(Tw):
            q_rad = SIGMA * F * A1 * (Tg ** 4 - Tw ** 4)
            return q_rad, convective_coefficient_W_m2K * A_tubes * (Tg - Tw)
        if R_wall is None:
            Tw = float(tube_wall_T_K)
        else:
            Tb = process["bulk_T_K"]
            Tw = brentq(lambda t: t - Tb - R_wall * (duty(t)[0] * alpha * A_cp / A1 + duty(t)[1]) / A_tubes, Tb, Tg, xtol=1e-8) \
                if Tg > Tb else Tb
        q_rad, q_conv = duty(Tw)
        return {"Tw": Tw, "em": em, "F": F, "q_rad": q_rad, "q_conv": q_conv}

    def residual(Tg):
        s = state(Tg)
        return released - setting - s["q_rad"] - s["q_conv"] - m * c.flue_sensible(Tg)
    T_ad = c.adiabatic_complete_T()
    lo = (tube_wall_T_K if tube_wall_T_K is not None else process["bulk_T_K"]) + 1.0
    if residual(lo) <= 0:
        raise ValueError("con ese fuego el gas no supera la temperatura de pared: no hay solución con transferencia al tubo")
    Tg = brentq(residual, lo, T_ad, xtol=1e-7)
    s = state(Tg)
    q_tubes_rad = s["q_rad"] * alpha * A_cp / A1
    q_shield = s["q_rad"] - q_tubes_rad
    Q_R = q_tubes_rad + s["q_conv"]
    flue_out = m * c.flue_sensible(Tg)
    balance = abs(released - setting - Q_R - q_shield - flue_out) / released
    em = s["em"]
    warnings = list(c.warnings)
    notes = []
    if gas_emissivity_override is None and exchange_factor_override is None and not em["in_fitted_range"]:
        warnings.append("Emisividad del gas fuera del intervalo del ajuste: " + "; ".join(em["out_of_range"]))
    if exchange_factor_override is not None:
        notes.append("Factor de intercambio F dado por el usuario (no calculado con la emisividad del gas).")
    elif gas_emissivity_override is not None:
        notes.append("Emisividad del gas dada por el usuario.")
    out = {
        "bridgewall_T_K": Tg, "effective_gas_T_K": Tg, "adiabatic_complete_combustion_T_K": T_ad,
        "radiant_duty_W": Q_R, "radiant_duty_by_radiation_W": q_tubes_rad, "radiant_duty_by_convection_W": s["q_conv"],
        "shield_radiant_duty_W": q_shield, "flue_gas_sensible_leaving_W": flue_out, "setting_loss_W": setting,
        "heat_release_LHV_W": m * c.LHV, "heat_release_HHV_W": m * c.HHV,
        "absorbed_fraction_of_LHV": (Q_R + q_shield) / (m * c.LHV),
        "average_flux_W_m2_tube_outside": Q_R / A_tubes, "tube_wall_T_K": s["Tw"],
        "tube_wall_from": "dato del usuario" if R_wall is None else
                          "T del fluido + flujo medio × (película interior + ensuciamiento + pared)",
        "radiation": {"gas_emissivity": em["emissivity"], "exchange_factor": s["F"], "alpha": alpha,
                      "alpha_from": "dato del usuario" if g.get("alpha") is not None else
                                    "factor de visión exacto de una fila + rerradiación del refractario",
                      "cold_plane_area_m2": A_cp, "equivalent_cold_plane_area_m2": A1, "envelope_area_m2": AT,
                      "refractory_to_sink_area_ratio": (AT - A1) / A1, "mean_beam_length_m": beam,
                      "pL_atm_m": em.get("pL_atm_m"), "p_H2O_Pa": p_w, "p_CO2_Pa": p_c, "tube_emissivity": tube_emissivity,
                      "gas_emissivity_source": em.get("source")},
        "tube_area_m2": A_tubes, "fuel_kg_s": m, "flue_gas_kg_s": m * c.flue_mass, "excess_air_fraction": c.excess_air,
        "evidence": {"energy_balance_error": balance, "verified_internal": balance < 1e-9, "verified": balance < 1e-9,
                     "reasons": [] if balance < 1e-9 else [f"el balance de energía del hogar no cierra ({balance:.2e})"],
                     "validated_against_literature": RADIANT_VALIDATION},
        "warnings": warnings,
        "assumptions": list(c.assumptions) + notes + [
            "Hogar de una sola zona de gas bien mezclada: la temperatura efectiva del gas es la de salida (pared divisoria). En "
            "hogares altos y estrechos o con quemadores de pared la efectiva puede ser bastante mayor.",
            "Gas gris a la temperatura del gas (misma emisividad para emitir y absorber), tubos grises, refractario "
            "radiativamente adiabático, recinto «moteado».",
            "Longitud media de haz 3.6 V/A salvo que se dé mean_beam_length_m.",
            "Sin llama luminosa ni hollín: solo radiación de CO2 y H2O.",
            "La pérdida por paredes dada se carga entera al hogar."],
        "not_included": ["perfil de flujo de calor a lo largo y alrededor del tubo (flujo pico)", "zonas múltiples",
                         "llama luminosa y partículas", "caída de presión y tiro", "tubos a dos caras o doble fila (da alpha)"],
        "method": {"name": "una zona de gas bien mezclada (Lobo-Evans / Hottel), factor de intercambio de red de radiación",
                   "module": "packages/energy/fired_heater.py", "property_data": CB.SOURCE},
    }
    return out


def radiant_sizing(required_radiant_duty_W: float, allowable_average_flux_W_m2: float, tube_outside_diameter_m: float,
                   tube_effective_length_m: float, tube_pitch_m: float) -> dict:
    """Tube surface for a radiant duty at an allowable average flux given by the user (outside tube area basis)."""
    for v, nme in ((required_radiant_duty_W, "required_radiant_duty_W"), (allowable_average_flux_W_m2,
                   "allowable_average_flux_W_m2"), (tube_outside_diameter_m, "tube_outside_diameter_m"),
                   (tube_effective_length_m, "tube_effective_length_m"), (tube_pitch_m, "tube_pitch_m")):
        if v is None or v <= 0:
            raise ValueError(f"{nme}: dato obligatorio y positivo (el flujo admisible lo fija el usuario, no hay valor típico)")
    area = required_radiant_duty_W / allowable_average_flux_W_m2
    per_tube = math.pi * tube_outside_diameter_m * tube_effective_length_m
    n = math.ceil(area / per_tube - 1e-9)
    row = tube_row_alpha(tube_pitch_m, tube_outside_diameter_m)
    return {"tube_area_required_m2": area, "area_per_tube_m2": per_tube, "n_tubes": n, "tube_area_installed_m2": n * per_tube,
            "average_flux_installed_W_m2": required_radiant_duty_W / (n * per_tube),
            "cold_plane_area_m2": n * tube_pitch_m * tube_effective_length_m, "alpha": row["alpha"],
            "equivalent_cold_plane_area_m2": row["alpha"] * n * tube_pitch_m * tube_effective_length_m,
            "next_step": "Con el número de tubos define el hogar (área de envolvente y volumen) y evalúalo con mode='rating': "
                         "el balance dirá si ese fuego entrega la carga y a qué temperatura de pared divisoria.",
            "assumptions": ["El flujo medio admisible es dato del usuario; el flujo pico (circunferencial y longitudinal) no "
                            "se calcula."]}


@lru_cache(maxsize=1)
def _transport_gas():
    return ct.Solution("gri30.yaml")


def _gas_properties(c: CB.Combustion, T_K: float, P_Pa: float) -> dict:
    gas = _transport_gas()
    x = {"CO2": c.flue.get("CO2", 0.0) + c.flue.get("SO2", 0.0), "H2O": c.flue.get("H2O", 0.0), "N2": c.flue.get("N2", 0.0),
         "O2": c.flue.get("O2", 0.0), "AR": c.flue.get("Ar", 0.0) + c.flue.get("He", 0.0), "CO": c.flue.get("CO", 0.0)}
    gas.TPX = T_K, P_Pa, x
    mu, k, cp = gas.viscosity, gas.thermal_conductivity, gas.cp_mass
    return {"mu": mu, "k": k, "cp": cp, "Pr": cp * mu / k}


BANK_KEYS = ("tube_outside_diameter_m", "tube_inside_diameter_m", "tube_effective_length_m", "tubes_per_row", "n_rows",
             "transverse_pitch_m", "longitudinal_pitch_m", "arrangement", "wall_conductivity_W_mK", "outside_fouling_m2K_W",
             "tube_emissivity")
CONV_PROCESS_KEYS = ("inlet_T_K", "inside_coefficient_W_m2K", "inside_fouling_m2K_W")


def convection_section(fuel: dict, air: dict, pressure_Pa: float, fuel_flow_kg_s: float, gas_inlet_T_K: float, bank: dict,
                       process: dict, excess_air_fraction: float | None = None, flue_O2: dict | None = None,
                       flue_CO_dry_mole_fraction: float | None = None, gas_radiation: bool = True) -> dict:
    """Convection bank, row by row, flue gas in cross flow and the process fluid counter-current overall.

    bank.tube_type: 'bare' (default; Žukauskas) or 'solid_fin' / 'serrated_fin' (revised ESCOA, Ganapathy 2015 app. E: fin
    efficiency, h_o = h_c + h_radiation, U on the total outside area with the outside fouling after the fin effectiveness) or
    'studded' (Esteem Projects stud correlation, gas radiation inside the correlation; see `stud_coefficients`)."""
    c = CB.solve_combustion(fuel, air, pressure_Pa, excess_air_fraction, flue_O2, flue_CO_dry_mole_fraction)
    if fuel_flow_kg_s is None or fuel_flow_kg_s <= 0 or gas_inlet_T_K is None:
        raise ValueError("da fuel_flow_kg_s y gas_inlet_T_K (temperatura del gas que entra al banco; la da fired_heater_radiant)")
    b, p = bank or {}, process or {}
    _need(b, BANK_KEYS, "bank")
    _need(p, CONV_PROCESS_KEYS, "process")
    isothermal = bool(p.get("isothermal"))
    if not isothermal and p.get("capacity_rate_W_K") is None:
        raise ValueError("process: da capacity_rate_W_K (flujo × cp del fluido) o isothermal=True (ebullición a T constante)")
    if b["arrangement"] not in ("staggered", "inline"):
        raise ValueError("bank.arrangement: 'staggered' o 'inline'")
    Do, Di, L = b["tube_outside_diameter_m"], b["tube_inside_diameter_m"], b["tube_effective_length_m"]
    n_t, n_r, St, Sl = int(b["tubes_per_row"]), int(b["n_rows"]), b["transverse_pitch_m"], b["longitudinal_pitch_m"]
    if not (0 < Di < Do < St) or Sl <= 0 or n_t < 1 or n_r < 1:
        raise ValueError("bank: se necesita Di < Do < paso transversal, paso longitudinal > 0 y al menos un tubo y una fila")
    tube_type = b.get("tube_type") or "bare"
    if tube_type not in ("bare", "studded") + FIN_TYPES:
        raise ValueError("bank.tube_type: 'bare', 'solid_fin', 'serrated_fin' o 'studded' (tubos con pernos)")
    finned, studded = tube_type in FIN_TYPES, tube_type == "studded"
    width = b.get("duct_width_m") or n_t * St
    if studded:
        _need(b, STUD_KEYS, "bank (tubos con pernos)")
        geo = studded_tube_geometry(Do, b["stud_length_m"], b["stud_diameter_m"], b["studs_per_plane"], b["stud_planes_per_m"])
        if Do + 2 * b["stud_length_m"] >= St:
            raise ValueError("bank: el diámetro exterior de los pernos (d + 2 L_perno) no cabe en el paso transversal")
        gap = width - n_t * geo["obstruction_m"]                 # free area of the Esteem stud script
        area_ratio, beam_default = geo["area_ratio_to_bare"], bank_beam_length(St, Sl, Do)
    elif finned:
        _need(b, FIN_KEYS + (("fin_segment_width_m",) if tube_type == "serrated_fin" else ()), "bank (tubos aletados)")
        hf, bf, kf = b["fin_height_m"], b["fin_thickness_m"], b["fin_conductivity_W_mK"]
        geo = finned_tube_geometry(Do, tube_type, hf, bf, b["fins_per_m"], b.get("fin_segment_width_m"))
        if Do + 2 * hf >= St:
            raise ValueError("bank: el diámetro exterior de la aleta (d + 2 h) no cabe en el paso transversal")
        gap = width - n_t * geo["obstruction_m"]                 # ESCOA: free area between obstructions (Ganapathy E.3)
        area_ratio = geo["area_ratio_to_bare"]
        beam_default = finned_bank_beam_length(St, Sl, Do, hf)
    else:
        gap = bank_free_width(width, n_t, Do, St, Sl, b["arrangement"])
        area_ratio, beam_default = 1.0, bank_beam_length(St, Sl, Do)
    if gap <= 0:
        raise ValueError("bank: no queda área libre entre tubos")
    m, m_gas = float(fuel_flow_kg_s), float(fuel_flow_kg_s) * c.flue_mass
    G_max = m_gas / (gap * L)
    A_row = n_t * math.pi * Do * L * area_ratio                 # total outside area of a row
    R_in = (Do / (Di * p["inside_coefficient_W_m2K"]) + p["inside_fouling_m2K_W"] * Do / Di +
            Do * math.log(Do / Di) / (2.0 * b["wall_conductivity_W_mK"])) * area_ratio
    wet = c.flue_mol
    p_w, p_c = c.flue.get("H2O", 0.0) / wet * pressure_Pa, c.flue.get("CO2", 0.0) / wet * pressure_Pa
    beam = b.get("mean_beam_length_m") or beam_default
    if beam <= 0:
        raise ValueError("bank: la longitud media de haz sale <= 0 (pasos muy cerrados para esas aletas); da mean_beam_length_m")
    fin_iter_failed = []
    C_p = None if isothermal else float(p["capacity_rate_W_K"])
    T_in_proc = float(p["inlet_T_K"])
    flags = set()

    def row_solve(Tg_in: float, Tp_hot: float) -> dict:
        def calc(Tg_out):
            Tg = 0.5 * (Tg_in + Tg_out)
            Q = m * (c.flue_sensible(Tg_in) - c.flue_sensible(Tg_out))
            Tp_cold = Tp_hot if isothermal else Tp_hot - Q / C_p
            Tp = 0.5 * (Tp_hot + Tp_cold)
            pr = _gas_properties(c, Tg, pressure_Pa)
            Tw = min(max(Tp + Q / A_row * R_in, Tp), Tg)
            Re = G_max * Do / pr["mu"]
            eg = None
            if gas_radiation and not studded and Tg - Tw > 1e-6:
                e = gas_emissivity(Tg, p_w, p_c, beam, pressure_Pa)
                eg = e["emissivity"]
                if not e["in_fitted_range"]:
                    flags.add("Emisividad del gas fuera del intervalo del ajuste WSGG (600-2400 K, 0.001-10 atm·m, 1 atm) en "
                              "alguna fila: la parte radiante de esas filas es una extrapolación.")
            out = {"Q": Q, "Tg_out": Tg_out, "Tp_cold": Tp_cold, "Tw": Tw, "Re": Re, "eg": eg, "Tg": Tg, "Tp": Tp}
            if studded:
                sc = stud_coefficients(G_max, Tg, Do, b["stud_length_m"], b["stud_diameter_m"], b["stud_conductivity_W_mK"],
                                       b["outside_fouling_m2K_W"], geo, b["arrangement"], Sl)
                U = 1.0 / (1.0 / sc["h_e"] + R_in)                 # outside fouling is inside h_so (source)
                out.update(h_c=sc["h_e"], h_r=0.0, U=U, stud=sc, E=sc["E"], zk={"notes": []})
            elif not finned:
                zk = zukauskas_bank_nusselt(Re, pr["Pr"], _gas_properties(c, Tw, pressure_Pa)["Pr"], n_r, St, Sl,
                                            b["arrangement"])
                h_c = zk["Nu"] * pr["k"] / Do
                h_r = gas_radiation_coefficient(Tg, Tw, eg, b["tube_emissivity"]) if eg is not None else 0.0
                U = 1.0 / (1.0 / (h_c + h_r) + b["outside_fouling_m2K_W"] + R_in)
                out.update(zk=zk, h_c=h_c, h_r=h_r, U=U)
            else:
                ec = escoa_colburn(Re, n_r, St, Sl, Do, hf, geo["fin_gap_m"], tube_type, b["arrangement"])
                base = ec["j_geometric"] * G_max * pr["cp"] * pr["Pr"] ** -0.67
                Tf = Tw
                for _ in range(60):                        # mean fin temperature <-> h_c, h_r, fin efficiency
                    h_c = base * (Tg / Tf) ** ec["temperature_exponent"]
                    h_r = gas_radiation_coefficient(Tg, Tf, eg, b["tube_emissivity"]) if eg is not None and Tg - Tf > 1e-6 \
                        else 0.0
                    fe = fin_efficiency(tube_type, h_c + h_r, kf, Do, hf, bf, b.get("fin_segment_width_m"))
                    tip = Tg - (Tg - Tw) * fe["tip_ratio"]
                    Tf_new = 0.5 * (Tw + tip)
                    if abs(Tf_new - Tf) < 1e-6:
                        break
                    Tf = Tf_new
                else:
                    fin_iter_failed.append(Tg)
                eta = 1.0 - (1.0 - fe["E"]) * geo["fin_area_m2_per_m"] / geo["total_area_m2_per_m"]
                U = 1.0 / (1.0 / (eta * (h_c + h_r)) + b["outside_fouling_m2K_W"] + R_in)
                out.update(h_c=h_c, h_r=h_r, U=U, escoa=ec, E=fe["E"], eta=eta, Tf=Tf, Ttip=tip, zk={"notes": []})
            out["Q_ua"] = out["U"] * A_row * (Tg - Tp)
            return out
        lo = max(Tp_hot, 300.0)
        f = lambda t: calc(t)["Q"] - calc(t)["Q_ua"]      # noqa: E731
        if Tg_in <= Tp_hot or f(Tg_in - 1e-9) >= 0:
            return calc(Tg_in)
        t_lo = min(lo, Tg_in - 1e-6)
        if f(t_lo) < 0:                                    # the gas would cool below the fluid: pinch in this row
            return calc(t_lo)
        return calc(brentq(f, t_lo, Tg_in, xtol=1e-7))

    def march(Tp_hot_end: float):
        rows, Tg, Tp = [], float(gas_inlet_T_K), Tp_hot_end
        for j in range(n_r):
            r = row_solve(Tg, Tp)
            rows.append(dict(r, row=j + 1, Tg_in=Tg, Tp_hot=Tp))
            Tg, Tp = r["Tg_out"], r["Tp_cold"]
        return rows, Tg, Tp

    if isothermal:
        rows, Tg_out, _ = march(T_in_proc)
    else:
        g = lambda t: march(t)[2] - T_in_proc             # noqa: E731
        hi = float(gas_inlet_T_K)
        if g(T_in_proc) >= 0 or g(hi) <= 0:
            raise ValueError("no hay solución: revisa que el gas entre más caliente que el fluido y la capacidad del fluido")
        rows, Tg_out, _ = march(brentq(g, T_in_proc, hi, xtol=1e-7))
    Q = sum(r["Q"] for r in rows)
    Q_gas = m * (c.flue_sensible(gas_inlet_T_K) - c.flue_sensible(Tg_out))
    rad_share = sum(r["Q"] * r["h_r"] / (r["h_c"] + r["h_r"]) for r in rows) / Q if Q > 0 else 0.0
    closure = abs(Q - Q_gas) / max(Q_gas, 1e-12)
    closed = abs(Q - Q_gas) <= 1e-9 * max(Q_gas, 1.0)
    Re_all = [r["Re"] for r in rows]
    warnings = list(c.warnings) + sorted(flags)
    for note in sorted({n for r in rows for n in r["zk"]["notes"]}):
        warnings.append(f"Convección del banco (Reynolds {min(Re_all):.0f}-{max(Re_all):.0f}): {note}.")
    reasons = [] if closed else [f"el calor al proceso no iguala al que cede el gas ({closure:.2e})"]
    if fin_iter_failed:
        reasons.append(f"la temperatura media de aleta no convergió en {len(fin_iter_failed)} evaluaciones de fila")
    if finned and b["arrangement"] == "staggered":
        warnings.append("Aletas en tresbolillo: las constantes ESCOA revisadas no tienen ejemplo resuelto leído; contra el "
                        "ejemplo de Esteem Projects (versión anterior de ESCOA) h_c difiere -7 %. Trata h_c con ±10 %.")
    if finned:
        warnings.append("La fuente leída (Ganapathy 2015) no publica el intervalo de Reynolds ni de geometría de la "
                        f"correlación ESCOA; Reynolds del banco {min(Re_all):.0f}-{max(Re_all):.0f}.")
    import CoolProp.CoolProp as CP
    y_w = c.flue.get("H2O", 0.0) / wet
    dew = CP.PropsSI("T", "P", y_w * pressure_Pa, "Q", 0, "Water") if y_w * pressure_Pa > 700 else None
    if dew is not None and min(r["Tw"] for r in rows) < dew:
        warnings.append(f"Alguna pared de tubo queda bajo el punto de rocío del agua ({dew:.1f} K): condensación y corrosión.")
    table = []
    for r in rows:
        row = {"row": r["row"], "gas_in_K": r["Tg_in"], "gas_out_K": r["Tg_out"], "process_K": r["Tp"], "tube_wall_K": r["Tw"],
               "duty_W": r["Q"], "flux_W_m2": r["Q"] / A_row, "h_convection_W_m2K": r["h_c"], "h_radiation_W_m2K": r["h_r"],
               "U_W_m2K": r["U"], "Reynolds": r["Re"], "gas_emissivity": r["eg"]}
        if finned:
            row.update(fin_efficiency=r["E"], fin_effectiveness=r["eta"], mean_fin_T_K=r["Tf"], fin_tip_T_K=r["Ttip"])
        if studded:
            row.update(stud_efficiency=r["E"], h_base_tube_W_m2K=r["stud"]["h_t"], h_stud_W_m2K=r["stud"]["h_so"])
        table.append(row)
    if studded:
        warnings.append("Tubos con pernos: correlación del tutorial de Esteem Projects sin ejemplo resuelto publicado que la "
                        "valide; la fuente no da su intervalo de validez. Carga con incertidumbre no medida.")
        X_max = max(r["stud"]["X"] for r in rows)
        if X_max > 1.2:
            warnings.append(f"Pernos con X = {X_max:.2f} > 1.2: la eficiencia de perno de la fuente queda por debajo de la "
                            "solución de aleta de pin (-9.5 % a X = 1.5, -26 % a X = 2); pernos largos o de baja conductividad.")
        if not gas_radiation:
            warnings.append("gas_radiation=False no tiene efecto con tubos con pernos: la radiación del gas va dentro de la "
                            "correlación empírica y no se puede separar.")
        surface = ("Tubos con pernos. Coeficiente exterior efectivo de la correlación de Esteem Projects (h_t del tubo base, h_s "
                   "del perno, ambos con la radiación del gas incluida, en función de la velocidad másica en el área libre y la "
                   "temperatura media del gas de la fila), eficiencia de perno E = 1.95/(e^X + e^-X), X = L_s (2 h_so/(k_s "
                   "D_s))^0.5, h_e = (h_so E A_perno + h_t A_tubo)/A_total y factor (d/P_l)^0.333 en línea; todo tal como lo "
                   "imprime la fuente (en unidades inglesas, convertido aquí a SI).")
        fin_note = ("Ensuciamiento exterior aplicado solo a los pernos (h_so = 1/(1/h_s + R_fo)), como en la fuente; U referido "
                    "al área exterior total: 1/U = 1/h_e + resistencias interiores × A_total/A_lisa. La E de la fuente ajusta "
                    "la aleta de pin con punta adiabática tanh(mL)/mL (±2.4 % hasta X = 1.2; -9.5 % a X = 1.5, -26 % a X = 2).")
        beam_note, corr, name = None, STUD_SOURCE, "banco de tubos con pernos fila por fila"
        validation = STUDDED_VALIDATION
        not_inc = ["temperatura de la punta de los pernos (la fuente no da su relación)",
                   "mezcla de tubos lisos, aletados y con pernos en un mismo banco (un tipo por llamada; encadena llamadas)"]
    elif finned:
        surface = (f"Tubos con aleta {'sólida' if tube_type == 'solid_fin' else 'segmentada'}. Convección exterior: "
                   "ESCOA revisada (Ganapathy 2015, ap. E, Tabla E.2) con el arreglo declarado, velocidad másica en el área "
                   "libre entre obstrucciones (d + 2 n b h), propiedades del gas a la temperatura media de la fila y "
                   "(T_gas/T_aleta)^x con la temperatura media de aleta (base + punta)/2, iterada.")
        fin_note = ("Eficiencia de aleta: " + ("solución de Bessel de la aleta anular con punta adiabática" if tube_type ==
                    "solid_fin" else "aleta recta tanh(mh)/mh con m del segmento") + " (Ganapathy ecs. E.9/E.13), con "
                    "h_o = h_c + h_radiación; U referido al área exterior total: 1/U = 1/(η h_o) + R_ensuciamiento_ext + "
                    "resistencias interiores × A_total/A_lisa.")
        beam_note = "Ganapathy ec. E.15 para tubos aletados"
        corr, name = ESCOA_SOURCE, "banco de tubos aletados fila por fila"
        validation = FINNED_VALIDATION
        not_inc = ["mezcla de tubos lisos, aletados y con pernos en un mismo banco (un tipo por llamada; encadena llamadas)"]
    else:
        surface = ("Tubos lisos. Convección exterior: Žukauskas 1972, ecs. (39)-(44), con el arreglo declarado, la velocidad en "
                   "el área mínima de paso y la corrección por número de filas del banco entero aplicada a todas las filas.")
        fin_note, beam_note, corr, name = None, "3.6 V/A del espacio entre tubos", ZUKAUSKAS_SOURCE, "banco de tubos lisos fila por fila"
        validation = CONVECTION_VALIDATION
        not_inc = ["tubos aletados o con pernos (usa bank.tube_type='solid_fin', 'serrated_fin' o 'studded')"]
    method = {"name": name, "module": "packages/energy/fired_heater.py", "property_data": CB.SOURCE,
              "convection_correlation": corr, "tube_type": tube_type}
    if finned:
        method.update(escoa_factors=rows[0]["escoa"], fin_area_m2_per_m=geo["fin_area_m2_per_m"],
                      total_area_m2_per_m=geo["total_area_m2_per_m"])
    elif studded:
        method.update(stud_factors_first_row={k: rows[0]["stud"][k] for k in ("h_t", "h_s", "h_so", "X", "E", "h_e")},
                      stud_area_m2_per_m=geo["stud_area_m2_per_m"], total_area_m2_per_m=geo["total_area_m2_per_m"])
    else:
        method.update(equations_used=sorted({r["zk"]["equation"] for r in rows}), row_factor=rows[0]["zk"]["row_factor"])
    return {
        "stack_T_K": Tg_out, "gas_inlet_T_K": float(gas_inlet_T_K), "duty_W": Q, "gas_radiation_share_of_duty": rad_share,
        "process_outlet_T_K": rows[0]["Tp_hot"], "process_inlet_T_K": T_in_proc,
        "max_tube_wall_T_K": max(r["Tw"] for r in rows), "outside_area_m2": n_r * A_row,
        "bare_tube_area_m2": n_r * n_t * math.pi * Do * L, "mass_velocity_kg_m2s": G_max,
        "max_fin_tip_T_K": max(r["Ttip"] for r in rows) if finned else None,
        "mean_beam_length_m": None if studded else beam, "flue_gas_kg_s": m_gas, "rows": table,
        "evidence": {"energy_balance_error": closure, "verified_internal": closed, "verified": closed and not fin_iter_failed,
                     "reasons": reasons, "validated_against_literature": validation},
        "warnings": warnings,
        "assumptions": list(c.assumptions) + [surface] + ([fin_note] if fin_note else []) + [
            "Propiedades de transporte del gas: Cantera, datos de GRI-Mech 3.0, promedio de mezcla (SO2 contado como CO2)"
            + (" (con pernos solo para el Reynolds informado: la correlación no las usa)." if studded else "."),
            "Radiación del gas entre tubos" + (": incluida en la correlación empírica de pernos (no WSGG)." if studded else
                                               ": gas gris WSGG con longitud de haz "
                                               + ("dada por el usuario (bank.mean_beam_length_m)" if b.get("mean_beam_length_m")
                                                  else beam_note)
                                               + " y recinto todo sumidero." if gas_radiation
                                               else " NO incluida (gas_radiation=False)."),
            "Fila por fila con temperaturas medias aritméticas; fluido de proceso en contracorriente global"
            + (" (isotérmico)." if isothermal else ", cp constante (capacity_rate_W_K)."),
            "Ancho del ducto = tubos por fila × paso transversal salvo que se dé duct_width_m."],
        "not_included": not_inc + ["radiación del hogar que escapa a las primeras filas (va como "
                                   "shield_cold_plane_area_m2 en la sección radiante)", "pérdida por paredes del banco",
                                   "caída de presión del gas y tiro", "rerradiación de las paredes del ducto (Kern recomienda "
                                   "el factor de Monrad; no se aplica)"],
        "method": method,
    }
