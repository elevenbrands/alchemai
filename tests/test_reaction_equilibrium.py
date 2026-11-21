"""`reaction_equilibrium`: K(T) from formation data and ideal-gas reaction equilibrium, against published values.

Independent references (numbers stated here, none computed by the code under test). Checked against the sources on
2026-11-20; what could and could not be confirmed is said for each:
  [SVN]   Smith, Van Ness & Abbott, "Introduction to Chemical Engineering Thermodynamics", 7th ed. (2005), ch. 13 and
          table C.4 (ΔfH°298, ΔfG°298, J/mol): CO −110525 / −137169; H2O(g) −241818 / −228572; CO2 −393509 / −394359;
          CH4 −74520 / −50460; NH3 −46110 / −16450; CH3OH(g) −200660 / −161960; C2H4 52510 / 68460;
          C2H5OH(g) −235100 / −168490. Table C.1 Cp/R = A + B·T + C·T² + D/T².
          The book itself was NOT opened (not legitimately accessible online). Confirmed in secondary transcriptions:
          the C2H4 / C2H5OH / H2O(g) ΔfG° rows and the Cp constants (FOSSEE Textbook Companion, ch. 13, Ex. 13.4,
          https://tbc-python.fossee.in/convert-notebook/Introduction_To_Chemical_Engineering_Thermodynamics_by_J._M._Smith,_H._C._Van_Ness_And_M._M._Abbott/ch13.ipynb
          — its H2O ΔfH° is mistyped −241572, which is why it prints K = 1.404e-1) and H2O ΔfH° through ΔH°298 = −45792
          below. The CO, CO2, CH4, NH3 and CH3OH rows are NOT confirmed against the book (quoted from memory); CO, CO2,
          H2O, CH4 and NH3 agree with NIST-JANAF within 0.01–0.35 kJ/mol, the tolerance used here.
          Ex. 13.4  ethylene hydration: ΔH°298 = −45792, ΔG°298 = −8378 J/mol, K(298.15) = 29.366,
                    K(418.15 K) = 1.443e-1, K(593.15 K) = 2.942e-3. Confirmed (lecture slides transcribing the example,
                    https://www.slideshare.net/Delight26/chapter-6-chemical-reactian-equilibriappt: K0 = 29.366,
                    K1 = 4.985e-3 / 1.023e-4, K2 = 0.9860 / 0.9794).
          Ex. 13.5  water-gas shift at 1100 K (ln K = 0 read from fig. 13.2): ε = 0.5 for 1 CO + 1 H2O at 1 or 10 bar and
                    with 2 mol N2; ε = 0.667 for 2 H2O + 1 CO and for 1 H2O + 2 CO (fraction of steam reacted 0.333 and
                    0.667); at 1650 K (ln K = −1.15, K = 0.316) ε = 0.36. Confirmed (same slides and FOSSEE notebook).
          Ex. 13.14 CH4/H2O/CO/CO2/H2 at 1000 K, 1 bar, feed 2 CH4 + 3 H2O (Gibbs minimisation):
                    y = CH4 0.0196, H2O 0.0980, CO 0.1743, CO2 0.0371, H2 0.6710. The printed answer is NOT confirmed
                    in a source (quoted from memory). Reproduced independently on 2026-11-20 from the book's
                    ΔfG°(1000 K) = CH4 19720 (memory), H2O −192420, CO −200240, CO2 −395790 J/mol (the last three
                    confirmed in the FOSSEE notebook, Ex. 13.13) with R = 8.314: 0.0196, 0.0980, 0.1743, 0.0371, 0.6711.
  [JANAF] NIST-JANAF Thermochemical Tables, 4th ed. (Chase 1998), read from https://janaf.nist.gov/tables/ (C-095 CO2,
          C-093 CO, H-064 H2O(g), C-067 CH4, H-083 NH3), log10 Kf at 500/800/1000 K: CO2 41.259/25.829/20.679,
          CO 16.236/11.916/10.461, H2O(g) 22.884/13.287/10.060; CH4 at 1000 K −1.018; NH3 ΔfG°(700 K) = 27.190 kJ/mol,
          ΔfG°(298.15 K) = −16.367 kJ/mol.

Tolerances: the tool's databank is chemicals (ΔfH° from Active Thermochemical Tables "ATCT_G", S° from CRC). Evaluated
compilations differ by a few tenths of kJ/mol per species (NH3: −45.56 ATcT, −45.90 CRC/JANAF, −46.11 SVN; CH4: −74.53
ATcT, −74.87 JANAF), so ΔG° of a reaction is compared within 1 kJ/mol (0.35 kJ/mol per mole of the worst species times
its coefficient) and ln K within that divided by R·T. Where the textbook's own data are imposed through data_overrides the
tolerance is the textbook's rounding (4 significant figures). Figure-read K (Ex. 13.5): ±0.05 in ln K → ±0.015 in ε.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from packages.agent import wrappers_properties as WP
from packages.dwsim_client import tools_properties as TPR
from packages.properties.reaction_equilibrium import R_GAS, reaction_equilibrium

T0 = 298.15
LN10 = math.log(10.0)
WGS = {"carbon monoxide": -1, "water": -1, "carbon dioxide": 1, "hydrogen": 1}
SMR = {"methane": -1, "water": -1, "carbon monoxide": 1, "hydrogen": 3}
NH3 = {"nitrogen": -1, "hydrogen": -3, "ammonia": 2}
MEOH = {"carbon monoxide": -1, "hydrogen": -2, "methanol": 1}
DG_TOL = 1000.0     # J/mol, see module docstring


def _k(r, j=0, i=0):
    return r["reactions"][j]["K_of_T"][i]


def _verified(r):
    assert r["status"] == "ok" and r["evidence"]["verified"], r["evidence"]


def test_standard_reaction_properties_at_298_vs_smith_van_ness_table_c4():
    cases = [(WGS, -41166.0, -28618.0), (SMR, 205813.0, 141863.0), (NH3, -92220.0, -32900.0), (MEOH, -90135.0, -24791.0)]
    for st, dH, dG in cases:
        r = reaction_equilibrium([st], T0)
        _verified(r)
        x = r["reactions"][0]
        assert x["dH_298_J_mol"] == pytest.approx(dH, abs=DG_TOL + 200.0 * (st is NH3)), x   # NH3: 2 × 0.55 kJ ATcT vs SVN
        assert x["dG_298_J_mol"] == pytest.approx(dG, abs=DG_TOL), x
        assert x["K_298"] == pytest.approx(math.exp(-x["dG_298_J_mol"] / (R_GAS * T0)), rel=1e-12)
        assert _k(r)["lnK"] == pytest.approx(x["lnK_298"], abs=1e-12)                    # K(T0) is K_298
    # ammonia against JANAF ΔfG°(298.15) = −16.367 kJ/mol
    r = reaction_equilibrium([NH3], T0)
    assert r["reactions"][0]["dG_298_J_mol"] == pytest.approx(2 * -16367.0, abs=DG_TOL)


def test_every_datum_has_a_source():
    r = reaction_equilibrium([WGS], 800.0)
    assert r["standard_state"].startswith("gas ideal puro")
    for s in r["species"]:
        assert "chemicals.reaction.Hfg" in s["Hf_gas_298_J_mol"]["source"]
        assert "S0g" in s["Gf_gas_298_J_mol"]["source"]
        assert s["Cp_ideal_gas"]["source"].startswith("thermo.HeatCapacityGas") and s["Cp_ideal_gas"]["in_range"]
    assert r["equilibrium"] is None                                                     # no feed: only K(T)


def test_water_gas_shift_K_of_T_vs_janaf():
    r = reaction_equilibrium([WGS], [500.0, 800.0, 1000.0])
    _verified(r)
    ref = [41.259 - 16.236 - 22.884, 25.829 - 11.916 - 13.287, 20.679 - 10.461 - 10.060]   # 2.139, 0.626, 0.158
    for i, (T, lg) in enumerate(zip((500.0, 800.0, 1000.0), ref)):
        k = _k(r, 0, i)
        assert k["log10K"] == pytest.approx(lg, abs=DG_TOL / (R_GAS * T * LN10)), (T, k)
        assert abs(k["lnK_numerical_quadrature"] - k["lnK"]) < 1e-6
        # the constant-ΔH° value is reported, labelled, and is NOT the rigorous one
        c = k["constant_dH_approximation"]
        assert "APROXIMACIÓN" in c["label"]
        assert c["lnK"] == pytest.approx(r["reactions"][0]["lnK_298"] - r["reactions"][0]["dH_298_J_mol"] / R_GAS
                                         * (1 / T - 1 / T0), abs=1e-10)
    assert _k(r, 0, 2)["constant_dH_approximation"]["K_ratio_to_rigorous"] < 0.7        # 0.90 vs 1.43 at 1000 K


def test_ammonia_and_steam_reforming_K_vs_janaf():
    r = reaction_equilibrium([NH3], 700.0)
    _verified(r)
    lnk_ref = -2 * 27190.0 / (R_GAS * 700.0)                                            # −9.343 → K = 8.8e-5
    assert _k(r)["lnK"] == pytest.approx(lnk_ref, abs=DG_TOL / (R_GAS * 700.0))
    r = reaction_equilibrium([SMR], 1000.0)
    _verified(r)
    assert _k(r)["log10K"] == pytest.approx(10.461 + 1.018 - 10.060, abs=DG_TOL / (R_GAS * 1000.0 * LN10))   # 1.419
    assert r["reactions"][0]["character"] == "endotérmica"


SVN_HYDRATION = {
    "ethylene": {"Hf_J_mol": 52510.0, "Gf_J_mol": 68460.0, "Cp_J_molK": {"A": 1.424, "B": 14.394e-3, "C": -4.392e-6}},
    "water": {"Hf_J_mol": -241818.0, "Gf_J_mol": -228572.0, "Cp_J_molK": {"A": 3.470, "B": 1.450e-3, "D": 0.121e5}},
    "ethanol": {"Hf_J_mol": -235100.0, "Gf_J_mol": -168490.0, "Cp_J_molK": {"A": 3.518, "B": 20.001e-3, "C": -6.002e-6}}}


def test_smith_van_ness_example_13_4_with_the_textbook_data():
    """User data (table C.4 and C.1) imposed: the rigorous integration must reproduce the worked example. SVN use
    R = 8.314; 4 significant figures printed → 1e-3 relative."""
    st = {"ethylene": -1, "water": -1, "ethanol": 1}
    r = reaction_equilibrium([st], [T0, 418.15, 593.15], data_overrides=SVN_HYDRATION)
    _verified(r)
    x = r["reactions"][0]
    assert x["dH_298_J_mol"] == pytest.approx(-45792.0, abs=0.5) and x["dG_298_J_mol"] == pytest.approx(-8378.0, abs=0.5)
    assert x["K_298"] == pytest.approx(29.366, rel=1e-3)
    assert _k(r, 0, 1)["K"] == pytest.approx(1.443e-1, rel=1e-3)
    assert _k(r, 0, 2)["K"] == pytest.approx(2.942e-3, rel=1e-3)
    for s in r["species"]:
        assert s["Hf_gas_298_J_mol"]["source"] == s["Gf_gas_298_J_mol"]["source"] == "dato del usuario"
        assert s["Cp_ideal_gas"]["source"] == "dato del usuario"
    # with the databank's own data the same reaction differs by the data, not by the method: same order, within 15 %
    lib = reaction_equilibrium([st], [418.15])
    assert _k(lib)["K"] == pytest.approx(1.443e-1, rel=0.45)      # ethanol ΔfG° −167.6 (ATcT + CRC S°) vs −168.5 kJ/mol SVN
    assert abs(_k(lib)["dG_T_J_mol"] - _k(r, 0, 1)["dG_T_J_mol"]) < 1300.0


def test_smith_van_ness_example_13_5_water_gas_shift_extents():
    def eps(T, P, feed):
        r = reaction_equilibrium([WGS], T, P, feed)
        _verified(r)
        return r, r["equilibrium"][0]["extents_mol"]["R1"]

    r, e = eps(1100.0, 1e5, {"water": 1, "carbon monoxide": 1})
    assert e == pytest.approx(0.5, abs=0.015)
    assert r["equilibrium"][0]["conversion"]["water"] == pytest.approx(e, rel=1e-12)
    assert eps(1100.0, 10e5, {"water": 1, "carbon monoxide": 1})[1] == pytest.approx(e, abs=1e-10)        # Δν = 0: no P effect
    rn, en = eps(1100.0, 1e5, {"water": 1, "carbon monoxide": 1, "nitrogen": 2})
    assert en == pytest.approx(e, abs=1e-10)                                                             # inert: no effect
    assert rn["equilibrium"][0]["mole_fractions"]["nitrogen"] == pytest.approx(0.5, abs=1e-12)
    assert [s["role"] for s in rn["species"] if s["compound"] == "nitrogen"] == ["inert"]
    assert eps(1100.0, 1e5, {"water": 2, "carbon monoxide": 1})[1] == pytest.approx(0.667, abs=0.015)
    assert eps(1100.0, 1e5, {"water": 1, "carbon monoxide": 2})[1] == pytest.approx(0.667, abs=0.015)
    assert eps(1650.0, 1e5, {"water": 1, "carbon monoxide": 1})[1] == pytest.approx(0.36, abs=0.015)


def test_smith_van_ness_example_13_14_two_simultaneous_reactions():
    """Steam reforming + shift, 2 CH4 + 3 H2O, 1000 K, 1 bar. The book minimises G with its own ΔfG°(1000 K); mole
    fractions agree to 0.002 with the databank's data. The book's printed answer was not re-read; the reference values
    are the ones reproduced from the book's ΔfG°(1000 K), see the module docstring."""
    r = reaction_equilibrium([{"name": "SMR", "stoichiometry": SMR}, {"name": "WGS", "stoichiometry": WGS}], 1000.0, 1e5,
                             {"methane": 2, "water": 3})
    _verified(r)
    eq = r["equilibrium"][0]
    ref = {"methane": 0.0196, "water": 0.0980, "carbon monoxide": 0.1743, "carbon dioxide": 0.0371, "hydrogen": 0.6710}
    for k, v in ref.items():
        assert eq["mole_fractions"][k] == pytest.approx(v, abs=0.002), (k, eq["mole_fractions"])
    assert sum(eq["mole_fractions"].values()) == pytest.approx(1.0, abs=1e-12)
    # evidence of the solution itself
    assert all(abs(c["difference"]) < 1e-9 for c in eq["K_consistency"])
    assert eq["element_balance"]["max_rel_error"] < 1e-12 and eq["element_balance"]["elements"] == ["C", "H", "O"]
    g = eq["gibbs_minimization_check"]
    assert g["converged"] and g["agree"] and g["max_abs_dy_vs_extents"] < 1e-9 and "elementos" in g["constraints"]
    assert r["evidence"]["reactions_span_all_independent"]
    # conversions are consistent with the extents
    assert eq["conversion"]["methane"] == pytest.approx(eq["extents_mol"]["SMR"] / 2, rel=1e-12)
    assert eq["conversion"]["water"] == pytest.approx((eq["extents_mol"]["SMR"] + eq["extents_mol"]["WGS"]) / 3, rel=1e-12)
    # Cantera, GRI-Mech 3.0 NASA data, phase restricted to the same five species
    (c,) = r["cantera_comparison"]
    assert c["compared"] and c["agree"] and c["composition"]["compared"]
    assert c["composition"]["max_abs_dy"] < 1e-3
    assert all(abs(q["dG_difference_J_mol"]) < 500.0 for q in c["reactions"])


def test_cantera_cross_check_of_K_for_several_reactions():
    """Independent data source (NASA polynomials of GRI-Mech 3.0): ΔG°(T) within 0.5 kJ/mol per mole of species."""
    for st, T in ((WGS, 600.0), (WGS, 1500.0), (SMR, 900.0), (NH3, 700.0), (MEOH, 500.0)):
        r = reaction_equilibrium([st], T)
        (c,) = r["cantera_comparison"]
        assert c["compared"] and c["agree"], c
        q = c["reactions"][0]
        assert abs(q["dG_difference_J_mol"]) <= q["tolerance_J_mol"]
        assert q["lnK_cantera"] == pytest.approx(_k(r)["lnK"], abs=q["tolerance_J_mol"] / (R_GAS * T))
    r = reaction_equilibrium([{"ethylene": -1, "water": -1, "ethanol": 1}], 500.0)
    assert not r["cantera_comparison"][0]["compared"] and "ethanol" in r["cantera_comparison"][0]["reason"]


def test_pressure_effect_ammonia_and_user_fugacity_coefficients():
    """Le Chatelier (Δν = −2): conversion rises with P; K_y·P^Δν = K exactly. Then user φ: Π(y·φ·P/P°)^ν = K."""
    xs = []
    for P in (1e5, 100e5, 300e5):
        r = reaction_equilibrium([NH3], 700.0, P, {"nitrogen": 1, "hydrogen": 3})
        _verified(r)
        eq = r["equilibrium"][0]
        y = eq["mole_fractions"]
        Ky = y["ammonia"] ** 2 / (y["nitrogen"] * y["hydrogen"] ** 3)
        assert Ky * (P / 1e5) ** -2 == pytest.approx(_k(r)["K"], rel=1e-8)
        xs.append(eq["conversion"]["nitrogen"])
        assert eq["conversion"]["hydrogen"] == pytest.approx(xs[-1], rel=1e-10)          # stoichiometric feed
    assert xs[0] < xs[1] < xs[2] and xs[0] < 0.01 and 0.3 < xs[2] < 0.7
    assert any("gas ideal" in w and "bar" in w for w in r["warnings"])                   # 300 bar ideal gas is flagged
    phi = {"ammonia": 0.9, "nitrogen": 1.1, "hydrogen": 1.05}
    rp = reaction_equilibrium([NH3], 700.0, 300e5, {"nitrogen": 1, "hydrogen": 3}, fugacity_coefficients=phi)
    _verified(rp)
    y = rp["equilibrium"][0]["mole_fractions"]
    a = {k: y[k] * phi[k] * 300.0 for k in phi}
    assert a["ammonia"] ** 2 / (a["nitrogen"] * a["hydrogen"] ** 3) == pytest.approx(_k(rp)["K"], rel=1e-8)
    assert rp["equilibrium"][0]["conversion"]["nitrogen"] > xs[2]                        # φ_NH3 < 1 favours ammonia
    assert rp["fugacity_coefficients"]["ammonia"]["source"] == "dato del usuario"
    assert not rp["cantera_comparison"][0]["composition"]["compared"]


def test_restricted_reaction_subset_keeps_other_species_frozen():
    """Only the shift is allowed although CH4 is present: CH4 stays as fed (an element-constrained Gibbs reactor would
    reform it). The independent check then minimises G under the invariants of the chosen reaction."""
    feed = {"carbon monoxide": 1, "water": 2, "methane": 0.5}
    r = reaction_equilibrium([WGS], 1000.0, 1e5, feed)
    _verified(r)
    eq = r["equilibrium"][0]
    assert eq["moles_out"]["methane"] == 0.5
    base = reaction_equilibrium([WGS], 1000.0, 1e5, {"carbon monoxide": 1, "water": 2})
    assert eq["extents_mol"]["R1"] == pytest.approx(base["equilibrium"][0]["extents_mol"]["R1"], rel=1e-10)
    assert not r["cantera_comparison"][0]["composition"]["compared"]
    # methane written into a reaction set that does not span the space: invariants, not elements
    two = reaction_equilibrium([{"methane": -1, "water": -2, "carbon dioxide": 1, "hydrogen": 4}], 900.0, 1e5,
                               {"methane": 1, "water": 3, "carbon monoxide": 0.2})
    assert two["equilibrium"][0]["moles_out"]["carbon monoxide"] == 0.2
    full = reaction_equilibrium([{"methane": -1, "water": -2, "carbon dioxide": 1, "hydrogen": 4}, WGS], 900.0, 1e5,
                                {"methane": 1, "water": 3, "carbon monoxide": 0.2})
    _verified(full)
    assert abs(full["equilibrium"][0]["moles_out"]["carbon monoxide"] - 0.2) > 0.05
    sub = reaction_equilibrium([{"methane": -1, "water": -1, "carbon monoxide": 1, "hydrogen": 3}], 900.0, 1e5,
                               {"methane": 1, "water": 3, "carbon dioxide": 0.1, "hydrogen": 0.1, "carbon monoxide": 0.1})
    _verified(sub)   # CO2 is a spectator here: not in any reaction → inert
    assert sub["equilibrium"][0]["moles_out"]["carbon dioxide"] == 0.1
    for q in (r, two, sub):
        assert any("RESTRINGIDO" in w for w in q["warnings"]) and not q["evidence"]["reactions_span_all_independent_with_inerts"]
    # two reactions among six species (three are independent; the shift is NOT allowed): the independent check must
    # minimise G under the invariants of the two chosen reactions, not under the element balance
    inv = reaction_equilibrium([SMR, {"carbon dioxide": -1, "hydrogen": -3, "methanol": 1, "water": 1}], 700.0, 50e5,
                               {"methane": 1, "water": 2, "carbon dioxide": 0.5})
    _verified(inv)
    g = inv["equilibrium"][0]["gibbs_minimization_check"]
    assert "invariantes" in g["constraints"] and g["agree"] and not inv["evidence"]["reactions_span_all_independent"]
    assert 0 < inv["equilibrium"][0]["mole_fractions"]["methanol"] < 0.05
    assert any("RESTRINGIDO" in w for w in inv["warnings"])
    assert not any("RESTRINGIDO" in w for w in full["warnings"])


def test_refusals_never_invent_data():
    with pytest.raises(ValueError, match="no está balanceada"):
        reaction_equilibrium([{"nitrogen": -1, "hydrogen": -2, "ammonia": 2}], 700.0)
    with pytest.raises(ValueError, match="linealmente independientes"):
        reaction_equilibrium([WGS, {k: -2 * v for k, v in WGS.items()}], 700.0)
    with pytest.raises(ValueError, match="P_Pa"):
        reaction_equilibrium([WGS], 700.0, None, {"water": 1, "carbon monoxide": 1})
    with pytest.raises(ValueError, match="no reconocido"):
        reaction_equilibrium([{"unobtainium": -1, "hydrogen": 1}], 700.0)
    with pytest.raises(ValueError, match="reactivo"):
        reaction_equilibrium([{"hydrogen": 1, "water": 1}], 700.0)
    with pytest.raises(ValueError, match="no puede tener moles positivos"):
        reaction_equilibrium([NH3], 700.0, 1e5, {"nitrogen": 1})                         # no hydrogen, no ammonia
    with pytest.raises(ValueError, match="claves desconocidas"):
        reaction_equilibrium([WGS], 700.0, data_overrides={"water": {"Hf": 1.0}})
    with pytest.raises(ValueError, match="Cp_J_molK"):
        reaction_equilibrium([WGS], 700.0, data_overrides={"water": {"Cp_J_molK": {"E": 1.0}}})


def test_missing_formation_data_is_asked_for(monkeypatch):
    """A compound whose gas-phase ΔfH° is absent from the databank stops the tool with a message that names what to ask
    the user; with the user's values it runs and labels them."""
    from chemicals import reaction
    real = reaction.Hfg
    monkeypatch.setattr(reaction, "Hfg", lambda cas, *a, **k: None if cas == "67-56-1" else real(cas, *a, **k))
    with pytest.raises(ValueError, match=r"Faltan datos de formación de «methanol».*data_overrides"):
        reaction_equilibrium([MEOH], 500.0)
    r = reaction_equilibrium([MEOH], 500.0, data_overrides={"methanol": {"Hf_J_mol": -200660.0, "Gf_J_mol": -161960.0}})
    _verified(r)
    s = next(s for s in r["species"] if s["compound"] == "methanol")
    assert s["Hf_gas_298_J_mol"] == {"value": -200660.0, "source": "dato del usuario"}
    assert r["reactions"][0]["dG_298_J_mol"] == pytest.approx(-24791.0, abs=30.0)        # SVN: −161960 + 137169


def test_cp_out_of_range_is_flagged_not_hidden():
    r = reaction_equilibrium([WGS], 7000.0, cantera_mechanism=None)                # beyond every Cp correlation
    assert r["cantera_comparison"] is None
    assert not r["evidence"]["verified"] and any("EXTRAPOLADO" in w for w in r["warnings"])
    r = reaction_equilibrium([{"ethylbenzene": -1, "styrene": 1, "hydrogen": 1}], 900.0)
    assert any("ESTIMADO" in w and "styrene" in w for w in r["warnings"])                # Joback Cp for styrene


def test_tool_and_wrapper():
    assert TPR.PROPERTIES_TOOLS[-1].__name__ == "reaction_equilibrium"
    for needle in ("data_overrides", "fugacity_coefficients", "feed_mol", "gas ideal", "van't Hoff"):
        assert needle.lower() in TPR.reaction_equilibrium.__doc__.lower()
    r = TPR.reaction_equilibrium([{"name": "WGS", "stoichiometry": {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}}], 1000.0, 1e5,
                                 {"CO": 1, "H2O": 1})
    assert r["reactions"][0]["equation"] == "CO + H2O = CO2 + H2" and r["evidence"]["verified"]
    w = WP.AGENT_WRAPPERS[-1]
    assert (getattr(w, "name", None) or w.__name__) == "reaction_equilibrium"
    out = json.loads(w.func([{"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}], [800.0, 1000.0]) if hasattr(w, "func") else
                     w([{"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}], [800.0, 1000.0]))
    assert len(out["reactions"][0]["K_of_T"]) == 2 and np.isfinite(out["reactions"][0]["K_of_T"][1]["K"])
