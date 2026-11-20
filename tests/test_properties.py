"""Validation of the `props` block: pure-compound properties with provenance (packages.properties.pure) and the flash
independent of DWSIM (packages.properties.flash), plus their agent tools.

Sources read on 2026-11-14 (pages and PDFs downloaded that day; numbers below copied from them, none from memory):
- NIST Chemistry WebBook, SRD 69, https://webbook.nist.gov/cgi/cbook.cgi?ID=<CAS>&Mask=1 (gas thermochemistry) and
  &Mask=4 (phase change: Tboil, Tfus, Tc, Pc, ΔvapH°, Antoine log10(P/bar) = A − B/(T + C)), for water C7732185,
  methane C74828, ethanol C64175, benzene C71432, n-hexane C110543, ammonia C7664417, carbon dioxide C124389,
  acetone C67641, acetic acid C64197, toluene C108883. When the WebBook lists several values the one used is its
  «AVG» or first «Review» row (e.g. acetone ΔfH°gas −218.5 kJ/mol, Wiberg et al. 1991; other rows −216.4 to −217.5).
- Kim, J. H.; Kim, M. S. (2005), Fluid Phase Equilib. 238, 13-19, CO2 + propane VLE 253-323 K, ThermoML file
  10.1016/j.fluid.2005.09.006.json of the NIST TRC ThermoML archive, https://trc.nist.gov/ThermoML/
- Hartanto, D.; Schuur, B.; Kiss, A. A.; de Haan, A. B. (2025), J. Chem. Eng. Data 70(10), 4091-4104,
  doi:10.1021/acs.jced.5c00445 (open-access PDF, TU Delft repository): Table 2 (n-hexane + ethanol VLE at 101.3 kPa)
  and Table 8 (NRTL τij = Aij + Bij/T for the pair; the paper's Table 9 reports RMSD T 0.38 K, y1 0.007 with its own
  extended Antoine constants).
- Kamihama et al. (2012), ethanol + water VLE at 101.3 kPa, as transcribed in the VLizard seed data
  https://github.com/Lemonexe/VLizard (appPy/seed_data/Kamihama2012.tsv). Secondary transcription: the paper itself
  was not opened.
- IUPAC Solubility Data Series vol. 15 «Alcohols with Water» (A. F. M. Barton, ed.; critical evaluation of
  1-butanol + water by G. T. Hefter, 1982), PDF SDS-15 from the IUPAC-NIST Solubility Database,
  https://srdata.nist.gov/solubility/ : recommended (R) mutual solubilities in g/100 g solution.
- IUPAC Solubility Data Series vol. 37 «Hydrocarbons with Water and Seawater, Part I: C5 to C7» (SDS-37, same
  site), Tables 2 and 3 for n-hexane + water (tentative values, 298 K: x_hexane in water 2.3e-6, x_water in hexane
  4.7e-4).
- Wikipedia «Azeotrope tables», https://en.wikipedia.org/wiki/Azeotrope_tables (wikitext), water + n-butanol:
  heteroazeotrope 92.4 °C at 1 atm, 55.5 wt % n-butanol (secondary compilation; weak source, used as such).
- CoolProp 8.0.0 (installed) as reference equation of state: IAPWS-95 for water, HEOS/GERG mixture for CO2 + propane.
"""
from __future__ import annotations

import json
import math
import statistics as st

import pytest

from packages.agent import wrappers_properties as WP
from packages.dwsim_client import tools_properties as TPR
from packages.properties import flash as FL
from packages.properties import pure as PU

# ---------------------------------------------------------------------------------------------------------------------
# NIST WebBook (read 2026-11-14). Tb, Tm, Tc [K], Pc [bar], ΔfH°gas and ΔvapH°(298) [kJ/mol], Antoine (A, B, C, Tmin, Tmax)
NIST = {
    "water": dict(Tb=373.17, Tc=647.0, Pc=220.64, Hfg=-241.826, ant=(4.6543, 1435.264, -64.848, 255.9, 373.0)),
    "methane": dict(Tb=111.0, Tc=190.6, Pc=46.1, Hfg=-74.6, ant=(3.9895, 443.028, -0.49, 90.99, 189.99)),
    "ethanol": dict(Tb=351.5, Tm=159.0, Tc=514.0, Pc=63.0, Hfg=-234.0, Hvap=42.3,
                    ant=(5.24677, 1598.673, -46.424, 292.77, 366.63)),
    "benzene": dict(Tb=353.3, Tm=278.64, Tc=562.0, Pc=48.9, Hfg=82.9, Hvap=33.9,
                    ant=(4.01814, 1203.835, -53.226, 287.70, 354.07)),
    "n-hexane": dict(Tb=341.9, Tm=178.0, Tc=507.6, Pc=30.2, Hfg=-167.2, Hvap=31.0,
                    ant=(4.00266, 1171.53, -48.784, 286.18, 342.69)),
    "ammonia": dict(Tm=194.95, Tc=405.4, Pc=113.0, Hfg=-45.94, ant=(4.86886, 1113.928, -10.409, 239.6, 371.5)),
    "carbon dioxide": dict(Tc=304.18, Pc=73.80, Hfg=-393.51),
    "acetone": dict(Tb=329.3, Tm=178.7, Tc=508.0, Pc=48.0, Hfg=-218.5, Hvap=31.27,
                    ant=(4.42448, 1312.253, -32.445, 259.16, 507.60)),
    "acetic acid": dict(Tb=391.2, Tm=289.6, Tc=593.0, Pc=57.86, Hfg=-433.0, Hvap=51.6,
                    ant=(4.68206, 1642.54, -39.764, 290.26, 391.01)),
    "toluene": dict(Tb=383.8, Tm=178.1, Tc=593.0, Pc=41.0, Hfg=50.1, Hvap=37.0,
                    ant=(4.07827, 1343.943, -53.773, 308.52, 384.66)),
}
# measured maximum relative deviations (2026-11-14) of the library vs NIST, with a small margin for library updates
NIST_BAND = {"Tb": 0.007, "Tm": 0.0065, "Tc": 0.0045, "Pc": 0.025, "Hfg": 0.012}

# Kim & Kim (2005), ThermoML: (x_CO2, P [kPa], y_CO2) at 273.15 and 303.15 K; pure Psat [kPa] (CO2, propane)
KIM = {273.15: [(0.02, 582.3, 0.173), (0.042, 697.1, 0.302), (0.085, 903.8, 0.459), (0.165, 1293.1, 0.615),
                (0.209, 1471.6, 0.673), (0.275, 1737, 0.724), (0.3, 1818, 0.738), (0.376, 2034.5, 0.781),
                (0.451, 2281.2, 0.808), (0.49, 2385.9, 0.823), (0.524, 2479.6, 0.834), (0.582, 2621.9, 0.844),
                (0.656, 2788.5, 0.868), (0.699, 2881.6, 0.88), (0.814, 3113.2, 0.915)],
       303.15: [(0.025, 1260.5, 0.134), (0.046, 1411, 0.224), (0.084, 1702.2, 0.339), (0.169, 2310.6, 0.493),
                (0.213, 2588, 0.554), (0.308, 3214.3, 0.637), (0.356, 3582.5, 0.67), (0.45, 4142.3, 0.717),
                (0.518, 4451.6, 0.747), (0.599, 4898.7, 0.776), (0.69, 5398.9, 0.812), (0.708, 5523, 0.816),
                (0.807, 6111.5, 0.866), (0.859, 6411.7, 0.89)]}
KIM_PURE = {273.15: (3478.5, 473.9), 303.15: (7206.2, 1079.6)}

# Hartanto et al. (2025) Table 2: (T [K], x_hexane, y_hexane) at 101.3 kPa; Table 8 NRTL parameters
HARTANTO = [(347.18, 0.015, 0.144), (345.39, 0.021, 0.197), (340.46, 0.064, 0.381), (338.61, 0.092, 0.451),
            (336.82, 0.116, 0.497), (335.90, 0.132, 0.529), (335.35, 0.141, 0.548), (334.26, 0.162, 0.567),
            (332.51, 0.252, 0.606), (331.39, 0.577, 0.645), (331.26, 0.687, 0.654), (331.52, 0.802, 0.665),
            (332.35, 0.895, 0.708), (334.07, 0.958, 0.765), (336.10, 0.978, 0.843), (339.32, 0.990, 0.939)]
HARTANTO_NRTL = [{"i": "n-hexane", "j": "ethanol", "a_ij": -10.342, "a_ji": -3.854, "b_ij_K": 4120.934, "b_ji_K": 1829.392,
                  "alpha": 0.47}]

# Kamihama et al. (2012) via VLizard: (T [K], x_ethanol, y_ethanol) at 101.3 kPa
KAMIHAMA = [(368.18, 0.018, 0.18), (360.5, 0.079, 0.418), (359.7, 0.09, 0.441), (357.27, 0.147, 0.51),
            (355.38, 0.244, 0.567),
            (354.55, 0.311, 0.589), (354.11, 0.353, 0.604), (353.45, 0.425, 0.627), (352.96, 0.488, 0.652),
            (352.39, 0.57, 0.688), (352.11, 0.616, 0.713), (351.82, 0.67, 0.736), (351.64, 0.711, 0.76),
            (351.37, 0.802, 0.819),
            (351.3, 0.835, 0.844), (351.27, 0.869, 0.873), (351.26, 0.901, 0.9), (351.26, 0.914, 0.912),
            (351.27, 0.922, 0.92),
            (351.29, 0.94, 0.937), (351.35, 0.972, 0.969)]

# SDS-15 recommended values: T [K], g 1-butanol / 100 g aqueous solution, g water / 100 g organic solution
SDS15 = [(293.15, 7.8, 20.0), (298.15, 7.4, 20.3), (303.15, 7.1, 20.6), (313.15, 6.6, 21.4), (333.15, 6.5, 23.7),
         (353.15, 7.0, 27.5)]
MW_BUOH, MW_W, MW_HEX = 74.1216, 18.01528, 86.17536
P_ATM = 101325.0


def _x(w: float, m1: float, m2: float) -> float:
    """Mole fraction of 1 from its mass percentage w in a binary with 2."""
    return (w / m1) / (w / m1 + (100.0 - w) / m2)


def _dumps(r: dict) -> None:
    json.dumps(r, default=float, ensure_ascii=False)


def _flash(*a, **k) -> dict:
    r = FL.independent_flash(*a, **k)
    _dumps(r)
    return r


# ---- compound_properties --------------------------------------------------------------------------------------------
def test_constants_against_nist_webbook():
    worst = {k: 0.0 for k in NIST_BAND}
    for name, ref in NIST.items():
        r = PU.compound_properties(name)
        _dumps(r)
        c = r["constants"]
        for k, key, scale in (("Tb", "Tb_K", 1.0), ("Tm", "Tm_K", 1.0), ("Tc", "Tc_K", 1.0), ("Pc", "Pc_Pa", 1e-5),
                              ("Hfg", "Hf_gas_J_mol", 1e-3)):
            if k in ref:
                v = c[key]["value"] * scale
                assert c[key]["method"] and c[key]["basis"] in ("data", "compilation"), (name, key, c[key])
                worst[k] = max(worst[k], abs(v / ref[k] - 1.0))
    # measured: Tb 0.60 % (methane, NIST avg 111 ± 2 K), Tm 0.62 % (toluene), Tc 0.39 % (acetic acid), Pc 2.24 %
    # (acetone; NIST avg 48.0 bar vs HEOS 46.92 bar), ΔfH°gas 1.11 % (acetone vs Wiberg 1991)
    for k, band in NIST_BAND.items():
        assert worst[k] <= band, (k, worst[k])
    assert worst["Pc"] > 0.02            # the acetone Pc deviation is real and stays documented


def test_vapour_pressure_and_hvap_against_nist():
    devs = []
    for name, ref in NIST.items():
        if "ant" in ref:
            A, B, C, lo, hi = ref["ant"]
            for T in (lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)):
                p = PU.compound_properties(name, T)["properties_at_T"]["Psat_Pa"]
                assert p["basis"] == "data" and p["in_range"]
                devs.append(abs(p["value"] / (1e5 * 10 ** (A - B / (T + C))) - 1.0))
    # measured max 2.22 % (methane 165 K, Antoine of the WebBook); water vs the 1947 Stull Antoine 1.78 %
    assert max(devs) < 0.025 and st.median(devs) < 0.005
    hv = {}
    for name, ref in NIST.items():
        if "Hvap" in ref:
            r = PU.compound_properties(name, 298.15)
            hv[name] = r["properties_at_T"]["Hvap_J_mol"]["value"] / 1e3 / ref["Hvap"] - 1.0
    # measured: within 2.81 % (toluene) for ethanol, benzene, n-hexane, acetone, toluene
    assert max(abs(v) for k, v in hv.items() if k != "acetic acid") < 0.03
    # acetic acid: DIPPR ΔHvap is to the real (dimerised) vapour, 23.4 vs ΔvapH° 51.6 kJ/mol: −55 %. Not hidden:
    assert hv["acetic acid"] < -0.5
    r = PU.compound_properties("acetic acid", 298.15)
    assert not r["evidence"]["verified"]
    assert any("se asocia" in w for w in r["warnings"])


def test_water_against_coolprop_iapws95():
    CP = pytest.importorskip("CoolProp.CoolProp")
    for T in (280.0, 300.0, 350.0, 400.0, 500.0, 600.0):
        r = PU.compound_properties("agua", T)
        _dumps(r)
        p = r["properties_at_T"]
        ps = CP.PropsSI("P", "T", T, "Q", 0, "Water")
        hv = (CP.PropsSI("H", "T", T, "Q", 1, "Water") - CP.PropsSI("H", "T", T, "Q", 0, "Water")) * 0.01801528
        assert abs(p["Psat_Pa"]["value"] / ps - 1) < 1e-5 and p["Psat_Pa"]["method"] == "IAPWS_PSAT"
        assert abs(p["Hvap_J_mol"]["value"] / hv - 1) < 1e-4                         # measured < 1e-5
        rho_l = CP.PropsSI("D", "T", T, "Q", 0, "Water")
        assert abs(p["rho_liquid_kg_m3"]["value"] / rho_l - 1) < 1.5e-3                # measured 0.08 % at 600 K
        assert r["evidence"]["verified"]
    mu = PU.compound_properties("water", 300.0)["properties_at_T"]["mu_liquid_Pa_s"]["value"]
    assert abs(mu / CP.PropsSI("V", "T", 300.0, "Q", 0, "Water") - 1) < 0.005
    # critical constants of the library are the reference-EOS ones (consistency, same source)
    for name, cpn in (("methane", "Methane"), ("ethanol", "Ethanol"), ("benzene", "Benzene"), ("toluene", "Toluene")):
        c = PU.compound_properties(name)["constants"]
        assert abs(c["Tc_K"]["value"] / CP.PropsSI("Tcrit", cpn) - 1) < 1e-5
        assert abs(c["Pc_Pa"]["value"] / CP.PropsSI("pcrit", cpn) - 1) < 3e-4


def test_provenance_consistency_and_states():
    r = PU.compound_properties("benzene", 350.0, 2e5, alternatives=True)
    _dumps(r)
    ev = r["evidence"]
    assert ev["verified"] and abs(ev["checks"]["psat_at_Tb_rel_dev"]["value"]) < 1e-3
    assert abs(ev["checks"]["clapeyron_Hvap_rel_dev"]["value"]) < 0.01          # measured −0.09 %
    assert r["state"]["phase"] == "liquid" and r["properties_at_T"]["rho_gas_kg_m3"]["value"] is None
    for key, p in r["properties_at_T"].items():
        if p.get("value") is not None:
            assert p["method"] and "in_range" in p, key
    assert isinstance(r["properties_at_T"]["Psat_Pa"].get("alternatives"), list)
    for k in ("warnings", "assumptions", "not_included", "method", "evidence"):
        assert k in r
    g = PU.compound_properties("methane", 300.0, 5e6)                              # supercritical T
    assert g["state"]["phase"] == "supercritical" and g["properties_at_T"]["Psat_Pa"]["value"] is None
    assert g["properties_at_T"]["rho_gas_kg_m3"]["method"] == "COOLPROP"
    s = PU.compound_properties("benzene", 250.0, P_ATM)                             # below Tm
    assert s["state"]["phase"] == "solid" and any("fusión" in w for w in s["warnings"])
    by_cas = PU.compound_properties("71-43-2")
    assert by_cas["identity"]["identified_by"] == "CAS" and by_cas["identity"]["CAS"] == "71-43-2"
    assert PU.compound_properties("etanol")["identity"]["CAS"] == "64-17-5"
    gf = g["constants"]["Gf_gas_J_mol"]                                          # identity ΔfG° = ΔfH° − T·ΔfS°
    hf = g["constants"]["Hf_gas_J_mol"]["value"]
    assert abs(gf["value"] - (hf - 298.15 * gf["Sf_gas_J_molK"])) < 1e-6 * abs(gf["value"])


def test_compound_refusals():
    with pytest.raises(ValueError, match="Compuesto no reconocido.*parecidos.*benzene"):
        PU.compound_properties("benzenee")
    with pytest.raises(ValueError, match="Falta 'compound'"):
        PU.compound_properties("")
    with pytest.raises(ValueError, match="sin T_K"):
        PU.compound_properties("water", None, 1e5)
    with pytest.raises(ValueError, match="positiva"):
        PU.compound_properties("water", -5.0)


# ---- Rachford-Rice and Raoult (analytical) ---------------------------------------------------------------------------
def test_rachford_rice_by_hand():
    z, K = [0.4, 0.6], [2.5, 0.4]
    beta_hand = -(z[0] * (K[0] - 1) + z[1] * (K[1] - 1)) / ((K[0] - 1) * (K[1] - 1))
    r = FL.rachford_rice(z, K)
    assert abs(r["phase_fraction"] - beta_hand) < 1e-13 and abs(r["residual"]) < 1e-13
    z3, K3 = [0.5, 0.3, 0.2], [2.0, 0.8, 0.1]
    r3 = FL.rachford_rice(z3, K3)
    b = r3["phase_fraction"]
    assert 0 < b < 1 and abs(sum(b * y + (1 - b) * x - zi for x, y, zi in zip(r3["x"], r3["y"], z3))) < 1e-14
    assert abs(sum(r3["x"]) - 1) < 1e-12 and abs(sum(r3["y"]) - 1) < 1e-12
    neg = FL.rachford_rice([0.1, 0.9], [1.5, 0.9])                                 # feed outside the two-phase region
    assert abs(neg["phase_fraction"] - (-0.8)) < 1e-12                             # by hand: −(0.05 − 0.09)/(0.5·(−0.1))
    with pytest.raises(ValueError):
        FL.rachford_rice([0.5, 0.5], [2.0, 1.5])


def test_raoult_bubble_dew_and_split_analytical():
    T = 350.0
    pb = PU.compound_properties("benzene", T)["properties_at_T"]["Psat_Pa"]["value"]
    pt = PU.compound_properties("toluene", T)["properties_at_T"]["Psat_Pa"]["value"]
    for z in (0.2, 0.5, 0.8):
        rb = _flash(["benzene", "toluene"], [z, 1 - z], "IDEAL", T_K=T, vapor_fraction=0.0)
        rd = _flash(["benzene", "toluene"], [z, 1 - z], "IDEAL", T_K=T, vapor_fraction=1.0)
        pbub, pdew = z * pb + (1 - z) * pt, 1 / (z / pb + (1 - z) / pt)
        assert abs(rb["P_Pa"] / pbub - 1) < 1e-10 and abs(rd["P_Pa"] / pdew - 1) < 1e-10
        assert abs(rb["phases"][0]["mole_fractions"]["benzene"] - z * pb / pbub) < 1e-10
        assert rb["evidence"]["verified"] and rd["evidence"]["verified"]
        P = 0.5 * (pbub + pdew)
        rm = _flash(["benzene", "toluene"], [z, 1 - z], "IDEAL", T_K=T, P_Pa=P)
        K1, K2 = pb / P, pt / P
        assert abs(rm["vapor_fraction"] - (-(z * (K1 - 1) + (1 - z) * (K2 - 1)) / ((K1 - 1) * (K2 - 1)))) < 1e-10
        ev = rm["evidence"]
        assert ev["verified"] and ev["material_balance_max_abs"] < 1e-12 and ev["rachford_rice_max_abs_residual"] < 1e-12
        assert abs(ev["rachford_rice_phase_fraction_difference"]) < 1e-10
        # NRTL with τ = 0 is Raoult
        rn = _flash(["benzene", "toluene"], [z, 1 - z], "NRTL", T_K=T, vapor_fraction=0.0,
                    activity_parameters=[{"i": "benzene", "j": "toluene", "b_ij_K": 0.0, "b_ji_K": 0.0, "alpha": 0.3}])
        assert abs(rn["P_Pa"] / pbub - 1) < 1e-10
    # monotonicity: bubble T rises with the heavy component; K symmetry V/L
    t = [_flash(["benzene", "toluene"], [z, 1 - z], "IDEAL", P_Pa=P_ATM, vapor_fraction=0.0)["T_K"] for z in (0.9, 0.5, 0.1)]
    assert t[0] < t[1] < t[2]


# ---- published VLE ---------------------------------------------------------------------------------------------------
def test_co2_propane_pr_library_kij_vs_kim_2005():
    dP, dy = [], []
    for T, pts in KIM.items():
        for x, P, y in pts:
            r = _flash(["carbon dioxide", "propane"], [x, 1 - x], "PR", T_K=T, vapor_fraction=0.0)
            assert r["evidence"]["verified"]
            dP.append(r["P_Pa"] / (P * 1e3) - 1)
            dy.append(r["phases"][0]["mole_fractions"]["carbon dioxide"] - y)
    assert r["binary_parameters"][0]["source"] == "library" and abs(r["binary_parameters"][0]["kij"] - 0.1315) < 1e-12
    # measured (29 points): |ΔP| ≤ 2.37 %, mean +0.69 %; |Δy| ≤ 0.0149
    assert max(abs(v) for v in dP) < 0.025 and abs(st.mean(dP)) < 0.01 and max(abs(v) for v in dy) < 0.016
    for T, (pco2, pc3) in KIM_PURE.items():                                        # pure-component ends: ≤ 0.19 %
        for n, pe in (("carbon dioxide", pco2), ("propane", pc3)):
            assert abs(PU.compound_properties(n, T)["properties_at_T"]["Psat_Pa"]["value"] / (pe * 1e3) - 1) < 0.0025
    # kij = 0 by absence: the warning is there and the answer is poor (measured |ΔP| up to 33 %)
    r0 = _flash(["carbon dioxide", "propane"], [0.3, 0.7], "PR", T_K=273.15, vapor_fraction=0.0, use_library_kij=False)
    assert r0["binary_parameters"][0]["source"] == "zero_by_absence" and any("kij(" in w for w in r0["warnings"])
    assert r0["P_Pa"] / 1818e3 - 1 < -0.15
    ru = _flash(["carbon dioxide", "propane"], [0.3, 0.7], "SRK", T_K=273.15, vapor_fraction=0.0,
                kij=[{"i": "carbon dioxide", "j": "propane", "kij": 0.1315}])             # PR's table value, given as user
    assert ru["binary_parameters"][0]["source"] == "user"


def test_co2_propane_against_coolprop_mixture():
    CP = pytest.importorskip("CoolProp.CoolProp")
    dP, dy = [], []
    for T, pts in KIM.items():
        for x, _, _ in pts[::2]:
            AS = CP.AbstractState("HEOS", "CO2&Propane")
            AS.set_mole_fractions([x, 1 - x])
            try:
                AS.update(CP.QT_INPUTS, 0.0, T)
            except ValueError:
                continue
            r = _flash(["carbon dioxide", "propane"], [x, 1 - x], "PR", T_K=T, vapor_fraction=0.0)
            dP.append(r["P_Pa"] / AS.p() - 1)
            dy.append(r["phases"][0]["mole_fractions"]["carbon dioxide"] - AS.mole_fractions_vapor()[0])
    # measured over the full set: |ΔP| ≤ 2.82 %, |Δy| ≤ 0.0168 (CoolProp itself is within 3.75 % of Kim's data)
    assert len(dP) >= 10 and max(abs(v) for v in dP) < 0.03 and max(abs(v) for v in dy) < 0.018


def _rmsd(v):
    return math.sqrt(st.mean(x * x for x in v))


def test_hexane_ethanol_vs_hartanto_2025():
    res = {}
    for label, kw in (("NRTL", dict(model="NRTL", activity_parameters=HARTANTO_NRTL)), ("UNIFAC", dict(model="UNIFAC")),
                      ("DORTMUND", dict(model="UNIFAC_DORTMUND")), ("IDEAL", dict(model="IDEAL"))):
        dT, dy = [], []
        for T, x, y in HARTANTO:
            r = _flash(["n-hexane", "ethanol"], [x, 1 - x], P_Pa=101300.0, vapor_fraction=0.0, **kw)
            assert r["evidence"]["verified"], (label, x)
            dT.append(r["T_K"] - T)
            dy.append(r["phases"][0]["mole_fractions"]["n-hexane"] - y)
        res[label] = (_rmsd(dT), max(abs(v) for v in dT), _rmsd(dy))
    # measured RMSD T / max |ΔT| / RMSD y: NRTL 0.80 K / 1.72 K / 0.0137 (paper: 0.38 K / 0.007 with its own Antoine);
    # UNIFAC 1.14 / 2.30 / 0.0198; Dortmund 0.61 / 1.77 / 0.0127; Raoult 11.9 K (no azeotrope)
    assert res["NRTL"][0] < 0.9 and res["NRTL"][1] < 1.9 and res["NRTL"][2] < 0.015
    assert res["UNIFAC"][0] < 1.25 and res["UNIFAC"][2] < 0.021
    assert res["DORTMUND"][0] < 0.7 and res["DORTMUND"][2] < 0.014
    assert res["IDEAL"][0] > 10.0


def test_ethanol_water_vs_kamihama_2012():
    for model, band_T, band_y in (("UNIFAC", 0.21, 0.0085), ("UNIFAC_DORTMUND", 0.2, 0.007)):
        dT, dy = [], []
        for T, x, y in KAMIHAMA:
            r = _flash(["ethanol", "water"], [x, 1 - x], model, P_Pa=101300.0, vapor_fraction=0.0)
            assert r["evidence"]["verified"]
            dT.append(r["T_K"] - T)
            dy.append(r["phases"][0]["mole_fractions"]["ethanol"] - y)
        # measured: UNIFAC RMSD T 0.194 K (max 0.47), RMSD y 0.0078; Dortmund 0.188 K (max 0.39), 0.0064
        assert _rmsd(dT) < band_T and max(abs(v) for v in dT) < 0.5 and _rmsd(dy) < band_y, model


def test_pr_kij0_ethanol_water_predicts_false_split_and_says_so():
    # PR with kij = 0 predicts a spurious liquid split for ethanol + water; thermo's VF solver then returns a state
    # whose material balance does not close, the independent check catches it and the own solver gives a consistent
    # three-phase bubble point (measured RMSD T 7.9 K over Kamihama's 21 points: the model is wrong, the numerics are not)
    r = _flash(["ethanol", "water"], [0.244, 0.756], "PR", P_Pa=101300.0, vapor_fraction=0.0)
    assert r["evidence"]["verified"] and r["solver"] == "own_three_phase_jump_resolution"
    assert r["liquid_liquid_split"] and abs(r["T_K"] - 355.38) > 5.0
    assert any("kij(ethanol, water) = 0" in w for w in r["warnings"])


# ---- liquid-liquid and VLLE ------------------------------------------------------------------------------------------
def test_water_butanol_lle_vs_sds15():
    dev = {}
    for model in ("UNIFAC", "UNIFAC_LLE"):
        for T, wb, ww in SDS15:
            r = _flash(["water", "1-butanol"], [0.8, 0.2], model, T_K=T, P_Pa=P_ATM)
            assert r["evidence"]["verified"] and r["phase_labels"] == ["L1", "L2"]
            assert r["evidence"]["stability"]["single_phase_feed"]["stable"] is False
            ph = {p["label"]: p for p in r["phases"]}
            dev[(model, T)] = (ph["L2"]["mole_fractions"]["1-butanol"] / _x(wb, MW_BUOH, MW_W) - 1,
                               ph["L1"]["mole_fractions"]["water"] / _x(ww, MW_W, MW_BUOH) - 1)
    # measured at 293-303 K: UNIFAC −6..+12 % (butanol in water), +1..+2 % (water in butanol); UNIFAC-LLE −12..+6 %, +4..+5 %
    for model in ("UNIFAC", "UNIFAC_LLE"):
        for T in (293.15, 298.15, 303.15):
            a, o = dev[(model, T)]
            assert abs(a) < 0.13 and abs(o) < 0.06, (model, T, a, o)
    # documented failure: neither model follows the minimum of the solubility of butanol in water (+47..+61 % at 333-353 K)
    assert dev[("UNIFAC", 353.15)][0] > 0.4 and dev[("UNIFAC_LLE", 353.15)][0] > 0.4


def test_water_butanol_heteroazeotrope_three_phase():
    # binary at fixed P: three phases only at one T, so P-VF, P-H and P-S fall on an isothermal jump
    r = _flash(["water", "1-butanol"], [0.8, 0.2], "UNIFAC", P_Pa=P_ATM, vapor_fraction=0.5)
    ev = r["evidence"]
    assert r["phase_labels"] == ["V", "L1", "L2"] and ev["verified"] and abs(ev["specification_residual"]) < 1e-9
    assert r["solver"] == "own_three_phase_jump_resolution" and ev["fugacity_equality_max_abs_dlnf"] < 1e-4
    y_w = {p["label"]: p for p in r["phases"]}["V"]["mole_fractions"]["water"]
    # Wikipedia azeotrope table: 92.4 °C, 55.5 wt % butanol (y_water 0.767). Measured: +0.61 K, Δy −0.008
    assert abs(r["T_K"] - 365.55) < 1.0 and abs(y_w - (1 - _x(55.5, MW_BUOH, MW_W))) < 0.01
    bub = _flash(["water", "1-butanol"], [0.8, 0.2], "UNIFAC", P_Pa=P_ATM, vapor_fraction=0.0)
    assert abs(bub["T_K"] - r["T_K"]) < 1e-5 and bub["evidence"]["verified"]
    rh = _flash(["water", "1-butanol"], [0.8, 0.2], "UNIFAC", P_Pa=P_ATM, H_J_mol=r["H_J_mol"])
    assert rh["evidence"]["verified"] and abs(rh["vapor_fraction"] - 0.5) < 1e-6
    rs = _flash(["water", "1-butanol"], [0.8, 0.2], "UNIFAC", P_Pa=P_ATM, S_J_molK=r["S_J_molK"])
    assert rs["evidence"]["verified"] and abs(rs["vapor_fraction"] - 0.5) < 1e-6


def test_hexane_water_pr_kij0_split_found_but_solubilities_wrong():
    r = _flash(["water", "n-hexane"], [0.5, 0.5], "PR", T_K=298.15, P_Pa=P_ATM)
    assert r["liquid_liquid_split"] and r["evidence"]["verified"]
    ph = {p["label"]: p for p in r["phases"]}
    # SDS-37 (tentative, 298 K): x_hexane in water 2.3e-6, x_water in hexane 4.7e-4. Measured: 4e-12 and 1.8e-2 —
    # NOT VALIDATED; the zero-kij warning is the honest answer
    assert ph["L2"]["mole_fractions"]["n-hexane"] < 1e-8 and ph["L1"]["mole_fractions"]["water"] > 10 * 4.7e-4
    assert any("kij(water, n-hexane) = 0" in w for w in r["warnings"])


def test_vlle_eos_roundtrips_ph_ps():
    r = _flash(["water", "n-hexane", "methane"], [0.4, 0.4, 0.2], "PR", T_K=320.0, P_Pa=2e5)
    ev = r["evidence"]
    assert r["phase_labels"] == ["V", "L1", "L2"] and ev["verified"]
    assert ev["material_balance_max_abs"] < 1e-8 and ev["fugacity_equality_max_abs_dlnf"] < 1e-6
    assert ev["stability"]["final_state_stable"] and ev["stability"]["second_liquid_found"]
    assert {b["source"] for b in r["binary_parameters"]} == {"library", "zero_by_absence"}
    for spec in ({"H_J_mol": r["H_J_mol"]}, {"S_J_molK": r["S_J_molK"]}):
        b = _flash(["water", "n-hexane", "methane"], [0.4, 0.4, 0.2], "PR", P_Pa=2e5, **spec)
        assert b["evidence"]["verified"] and abs(b["T_K"] - 320.0) < 1e-5
        assert abs(b["vapor_fraction"] - r["vapor_fraction"]) < 1e-6
    # an isenthalpic valve: same H at lower P must cool and vaporise more (monotonicity)
    v = _flash(["water", "n-hexane", "methane"], [0.4, 0.4, 0.2], "PR", P_Pa=1e5, H_J_mol=r["H_J_mol"])
    assert v["evidence"]["verified"] and v["T_K"] < 320.0 and v["vapor_fraction"] > r["vapor_fraction"]


# ---- refusals --------------------------------------------------------------------------------------------------------
def test_flash_refusals():
    c, z = ["benzene", "toluene"], [0.5, 0.5]
    with pytest.raises(ValueError, match="Falta 'model'"):
        FL.independent_flash(c, z, None, T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="no soportado"):
        FL.independent_flash(c, z, "VAN_LAAR", T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="suma"):
        FL.independent_flash(c, [0.5, 0.6], "PR", T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="exactamente dos"):
        FL.independent_flash(c, z, "PR", T_K=350.0, P_Pa=P_ATM, vapor_fraction=0.5)
    with pytest.raises(ValueError, match="exactamente dos"):
        FL.independent_flash(c, z, "PR", T_K=350.0)
    with pytest.raises(ValueError, match="necesita parámetros binarios"):
        FL.independent_flash(c, z, "NRTL", T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="alpha"):
        FL.independent_flash(c, z, "NRTL", T_K=350.0, P_Pa=P_ATM,
                             activity_parameters=[{"i": "benzene", "j": "toluene", "b_ij_K": 1.0, "b_ji_K": 1.0}])
    with pytest.raises(ValueError, match="kij solo aplica"):
        FL.independent_flash(c, z, "UNIFAC", T_K=350.0, P_Pa=P_ATM, kij=[{"i": "benzene", "j": "toluene", "kij": 0.0}])
    with pytest.raises(ValueError, match="IDEAL"):
        FL.independent_flash(c, z, "IDEAL", T_K=350.0, P_Pa=P_ATM,
                             activity_parameters=[{"i": "benzene", "j": "toluene", "b_ij_K": 0.0, "b_ji_K": 0.0}])
    with pytest.raises(ValueError, match="al menos dos"):
        FL.independent_flash(c, [1.0, 0.0], "PR", T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="no reconocido"):
        FL.independent_flash(["benzene", "unobtainium"], z, "PR", T_K=350.0, P_Pa=P_ATM)
    with pytest.raises(ValueError, match="vapor_fraction"):
        FL.independent_flash(c, z, "PR", T_K=350.0, vapor_fraction=1.5)


# ---- compare_flash and tools -----------------------------------------------------------------------------------------
def test_compare_flash_verdicts():
    c, z, kw = ["benzene", "toluene"], [0.5, 0.5], dict(T_K=369.0, P_Pa=P_ATM)
    ind = _flash(c, z, "PR", **kw)
    assert ind["phase_labels"] == ["V", "L"]
    ref = {"engine": "test", "phases": [{"type": p["type"], "fraction": p["mole_fraction_of_feed"],
                                         "mole_fractions": [p["mole_fractions"][n] for n in c]} for p in ind["phases"]]}
    same = FL.compare_flash(ref, c, z, "PR", **kw)
    _dumps(same)
    assert same["verdict"] == "agree" and same["max_abs_mole_fraction_difference"] < 1e-12
    shifted = json.loads(json.dumps(ref))
    shifted["phases"][0]["fraction"] += 0.05
    shifted["phases"][1]["fraction"] -= 0.05
    assert FL.compare_flash(shifted, c, z, "PR", **kw)["verdict"] == "disagree"
    lle = _flash(["water", "1-butanol"], [0.8, 0.2], "UNIFAC", T_K=298.15, P_Pa=P_ATM)
    one = {"engine": "no_stability",
           "phases": [{"type": "liquid", "fraction": 1.0, "mole_fractions": {"water": 0.8, "1-butanol": 0.2}}]}
    d = FL.compare_flash(one, ["water", "1-butanol"], [0.8, 0.2], "UNIFAC", T_K=298.15, P_Pa=P_ATM)
    assert lle["liquid_liquid_split"] and d["verdict"] == "phase_set_differs"
    with pytest.raises(ValueError, match="reference"):
        FL.compare_flash({}, c, z, "PR", **kw)


def test_tools_and_wrappers():
    assert [f.__name__ for f in TPR.PROPERTIES_TOOLS] == ["compound_properties", "independent_flash", "compare_flash",
                                                           "cubic_eos_volume", "reaction_equilibrium"]
    assert [getattr(w, "name", None) or w.__name__ for w in WP.AGENT_WRAPPERS] == [f.__name__ for f in TPR.PROPERTIES_TOOLS]
    r = TPR.compound_properties("toluene", 330.0)
    _dumps(r)
    assert r["properties_at_T"]["Psat_Pa"]["value"] > 0
    f = TPR.independent_flash(["benzene", "toluene"], [0.5, 0.5], "IDEAL", P_Pa=P_ATM, vapor_fraction=0.0)
    _dumps(f)
    assert f["status"] == "ok" and f["evidence"]["verified"]
    for k in ("warnings", "assumptions", "not_included", "method", "evidence"):
        assert k in f
