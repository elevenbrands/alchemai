"""Preliminary HAZOP: deterministic rules fire on the solved numbers (hydrates at the JT valve, runaway and
flammable mixture in combustion, relief case in the column) and the tool/Excel export work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.dwsim_client import tools as T  # noqa: E402
from packages.hazop.report import hazop_from_flowsheet, hazop_preliminary, to_excel  # noqa: E402


def _wet_gas():
    z = {"Methane": 0.84, "Ethane": 0.07, "Propane": 0.04, "N-butane": 0.02, "Carbon dioxide": 0.02, "Water": 0.01}
    fid = T.new_flowsheet("gas", list(z), "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300, pressure_Pa=50e5, molar_flow_mol_s=100, mole_fractions=z)
    for n in ("hp", "cold", "vap", "liq", "lp"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "compressor", "K-1", {"outlet_P_Pa": 60e5, "adiabatic_eff_pct": 75}); T.connect(fid, "K-1", feeds=["gas"], products=["hp"])
    T.add_unit(fid, "cooler", "E-1", {"outlet_T_K": 280}); T.connect(fid, "E-1", feeds=["hp"], products=["cold"])
    T.add_unit(fid, "separator", "V-1"); T.connect(fid, "V-1", feeds=["cold"], products=["vap", "liq"])
    T.add_unit(fid, "valve", "VLV-1", {"outlet_P_Pa": 15e5}); T.connect(fid, "VLV-1", feeds=["vap"], products=["lp"])
    assert T.solve(fid)["ok"]
    return fid


def test_wet_gas_hazop_rows(tmp_path):
    fid = _wet_gas()
    rows, g = hazop_from_flowsheet(T._s(fid)["fs"], "gas")
    assert len(rows) >= 15
    jt = [r for r in rows if r.node == "VLV-1" and "Joule-Thomson" in r.deviation]
    assert jt and any("hidratos" in c for c in jt[0].consequences), jt
    assert any(r.node == "V-1" and "blowby" in " ".join(r.consequences) for r in rows)
    assert any(r.node == "K-1" and "Surge" in " ".join(r.consequences) for r in rows)
    assert all(r.source == "rule" for r in rows)
    out = hazop_preliminary(fid, excel_path=str(tmp_path / "h.xlsx"))
    assert out["n_rows"] == len(rows) and (tmp_path / "h.xlsx").exists() and "| Nodo |" in out["markdown"]


def test_combustion_hazop_runaway_and_flammable_mixture():
    fid = T.new_flowsheet("comb", ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"], "PengRobinson")["flowsheet_id"]
    T.define_conversion_reaction(fid, "COMB", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane", 100, "Comb")
    T.add_material_stream(fid, "fuel", temperature_K=298.15, pressure_Pa=101325, molar_flow_mol_s=1, mole_fractions={"Methane": 1.0})
    T.add_material_stream(fid, "air", temperature_K=298.15, pressure_Pa=101325, molar_flow_mol_s=9.52, mole_fractions={"Oxygen": 0.21, "Nitrogen": 0.79})
    for n in ("mix", "flue", "liq"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "mixer", "MIX"); T.connect(fid, "MIX", feeds=["fuel", "air"], products=["mix"])
    T.add_unit(fid, "conversion_reactor", "R-1", {"reaction_set": "Comb", "mode": "isothermal"})
    T.connect(fid, "R-1", feeds=["mix"], products=["flue", "liq"], energy_feeds=["Q"])
    assert T.solve(fid)["ok"]
    rows, _ = hazop_from_flowsheet(T._s(fid)["fs"], "comb")
    assert any(r.node == "R-1" and "runaway" in r.deviation for r in rows)
    mix = [r for r in rows if r.node == "MIX" and "Flammable mixture" in r.deviation]
    assert mix and mix[0].severity == 5


def test_column_hazop_relief_case():
    fid = T.new_flowsheet("fug", ["Ethane", "Propane", "N-butane", "N-pentane"], "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "alim", pressure_Pa=17e5, vapor_fraction=0.0, molar_flow_mol_s=100,
                          mole_fractions={"Ethane": 0.05, "Propane": 0.45, "N-butane": 0.40, "N-pentane": 0.10})
    for n in ("d", "b"):
        T.add_material_stream(fid, n)
    for n in ("qc", "qr"):
        T.add_energy_stream(fid, n)
    T.add_unit(fid, "shortcut_column", "T-1", {"light_key": "Propane", "heavy_key": "N-butane", "lk_mole_frac_in_bottoms": 0.02,
                                              "hk_mole_frac_in_distillate": 0.02, "condenser_P_Pa": 16e5, "reboiler_P_Pa": 17.5e5, "reflux_ratio": 1.84})
    T.connect(fid, "T-1", feeds=["alim"], products=["d", "b"], energy_products=["qc"], energy_feeds=["qr"])
    assert T.solve(fid)["ok"]
    rows, _ = hazop_from_flowsheet(T._s(fid)["fs"], "fug")
    assert any("falla de condensador" in " ".join(r.safeguards).lower() for r in rows if r.node == "T-1")
