"""Gibbs reactor (element matrix defined from the species) and the shortcut-column phase check of the verifier."""
import math

import pytest

from packages.dwsim_client import tools as T


def _wgs(species=None, extra=()):
    comps = ["Carbon monoxide", "Water", "Carbon dioxide", "Hydrogen", *extra]
    fid = T.new_flowsheet("wgs", comps, "PengRobinson1978")["flowsheet_id"]
    flows = {"Carbon monoxide": 1.0, "Water": 1.0, **{c: 1.0 for c in extra}}
    T.add_material_stream(fid, "F", temperature_K=1000.0, pressure_Pa=101325.0, compound_molar_flows_mol_s=flows)
    for n in ("V", "L"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "gibbs_reactor", "R", {"mode": "isothermal", **({"species": species} if species else {})})
    T.connect(fid, "R", ["F"], ["V", "L"], energy_feeds=["Q"])
    return fid


def test_gibbs_reactor_reproduces_water_gas_shift_equilibrium():
    """CO + H2O = CO2 + H2 at 1000 K, 1 atm, equimolar feed. K(1000 K) = 1.44 from the JANAF tables (log10 Kf: CO2 20.680,
    CO 10.459, H2O 10.062 -> log10 K = 0.159), so the extent x of 1 mol satisfies (x/(1-x))^2 = K -> x = 0.545."""
    fid = _wgs()
    sol = T.solve(fid)
    assert sol["ok"], sol["solver_errors"]
    v = sol["streams"]["V"]
    x = v["molar_flow_mol_s"] * v["mole_fractions"]["Hydrogen"]
    assert x == pytest.approx(math.sqrt(1.44) / (1 + math.sqrt(1.44)), abs=0.015)
    assert v["molar_flow_mol_s"] == pytest.approx(2.0, rel=1e-6)          # equimolar reaction
    T.close_flowsheet(fid)


def test_gibbs_reactor_species_subset_leaves_inerts_untouched():
    fid = _wgs(species=["Carbon monoxide", "Water", "Carbon dioxide", "Hydrogen"], extra=("Nitrogen",))
    sol = T.solve(fid)
    assert sol["ok"], sol["solver_errors"]
    v = sol["streams"]["V"]
    assert v["molar_flow_mol_s"] * v["mole_fractions"]["Nitrogen"] == pytest.approx(1.0, rel=1e-6)
    T.close_flowsheet(fid)
    with pytest.raises(ValueError):
        f2 = T.new_flowsheet("x", ["Water", "Hydrogen"], "PengRobinson1978")["flowsheet_id"]
        T.add_unit(f2, "gibbs_reactor", "R", {"species": ["Water", "Argon"]})


def _pentane_splitter(package):
    k = 0.125998                                                           # lb mol/h -> mol/s
    fid = T.new_flowsheet("c5", ["Isobutane", "N-butane", "Isopentane", "N-pentane"], package)["flowsheet_id"]
    T.add_material_stream(fid, "F", pressure_Pa=248211.0, vapor_fraction=0.0, compound_molar_flows_mol_s={
        "Isobutane": 0.000665 * k, "N-butane": 4.947624 * k, "Isopentane": 395.9997 * k, "N-pentane": 499.6478 * k})
    for n in ("D", "B"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Qc")
    T.add_energy_stream(fid, "Qr")
    T.add_unit(fid, "shortcut_column", "C", {"light_key": "Isopentane", "heavy_key": "N-pentane",
                                             "lk_mole_frac_in_bottoms": 0.00794, "hk_mole_frac_in_distillate": 0.01243,
                                             "condenser_P_Pa": 137895.0, "reboiler_P_Pa": 137895.0, "reflux_ratio": 12.0})
    T.connect(fid, "C", ["F"], ["D", "B"], energy_products=["Qc"], energy_feeds=["Qr"])
    return T.solve(fid)


def test_shortcut_column_with_vapour_bottoms_is_not_ok():
    """With Chao-Seader DWSIM returns the bottoms of this column as vapour, which inflates the reboiler duty by the latent
    heat of the bottoms (the feed is hotter than both products, so Qr must be below Qc). The verifier must refuse it;
    with Peng-Robinson the bottoms are liquid, Qr < Qc and the unit is ok."""
    good = _pentane_splitter("PengRobinson1978")
    assert good["ok"] and good["streams"]["B"]["vapor_fraction"] < 0.01
    bad = _pentane_splitter("ChaoSeader")
    if bad["streams"]["B"]["vapor_fraction"] < 0.01:
        pytest.skip("DWSIM returned liquid bottoms with Chao-Seader in this version")
    assert bad["ok"] is False
    assert "vapor" in bad["units"][0]["error"] and "rehervidor" in bad["units"][0]["error"]
