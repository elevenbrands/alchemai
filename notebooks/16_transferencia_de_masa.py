"""Notebook 16 — mass transfer (bloque 22).

A) Kremser vs DWSIM's rigorous AbsorptionColumn: propane (2 %) absorbed from methane into n-decane at 10 bar, 300 K,
   Peng-Robinson; lean oil L = 30/60/120 mol/s, N = 4/6/10 stages (DWSIM rejects fewer than 4), gas 100 mol/s. Kremser uses K_propane from a DWSIM
   flash of gas + oil at 300 K / 10 bar and average L/V from the column's own end flows.
B) Packed height (Onda + Colburn) for the L = 120 mol/s case at 94 % recovery, illustrative packing data.
Usage: PYTHONPATH=. uv run python notebooks/16_transferencia_de_masa.py A|B
"""
import json
import sys

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.transport.mass_transfer import kremser_fraction_absorbed  # noqa: E402

COMPS = ["Methane", "Propane", "N-decane"]
GAS = {"Methane": 0.98, "Propane": 0.02, "N-decane": 0.0}
OIL = {"Methane": 0.0, "Propane": 0.0, "N-decane": 1.0}


def k_propane(L: float) -> float:
    fid = T.new_flowsheet("kflash", COMPS, "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=100.0, mole_fractions=GAS)
    T.add_material_stream(fid, "oil", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=L, mole_fractions=OIL)
    for n in ("mix", "v", "l"):
        T.add_material_stream(fid, n)
    T.add_unit(fid, "mixer", "M"); T.connect(fid, "M", feeds=["gas", "oil"], products=["mix"])
    T.add_unit(fid, "separator", "F"); T.connect(fid, "F", feeds=["mix"], products=["v", "l"])
    assert T.solve(fid)["solved"]
    K = T.read_stream(fid, "v")["mole_fractions"]["Propane"] / T.read_stream(fid, "l")["mole_fractions"]["Propane"]
    T.close_flowsheet(fid)
    return K


def column(L: float, N: int) -> dict:
    fid = T.new_flowsheet(f"abs_{L}_{N}", COMPS, "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=100.0, mole_fractions=GAS)
    T.add_material_stream(fid, "oil", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=L, mole_fractions=OIL)
    T.add_material_stream(fid, "top"); T.add_material_stream(fid, "bot")
    T.add_unit(fid, "absorption_column", "ABS", {"stages": N, "top_P_Pa": 10e5})
    T.connect(fid, "ABS", feeds=["oil", "gas"], products=["top", "bot"])
    r = T.solve(fid)
    top, bot = T.read_stream(fid, "top"), T.read_stream(fid, "bot")
    T.close_flowsheet(fid)
    absorbed = 1.0 - top["molar_flow_mol_s"] * top["mole_fractions"]["Propane"] / 2.0
    return {"solved": r["solved"], "ok": r["ok"], "frac_dwsim": absorbed, "V_top": top["molar_flow_mol_s"], "L_bot": bot["molar_flow_mol_s"],
            "T_top": top["T_K"], "T_bot": bot["T_K"]}


def part_b():
    """Packed height for the L = 120 mol/s propane absorber at 94 % recovery. Packing data and diffusion inputs are
    ILLUSTRATIVE (see MOCKS.md): 50 mm metal rings a_t = 105 m²/m³, σc = 0.075 N/m; Fuller volumes C 16.5, H 1.98
    (Fuller et al. 1966 table as reproduced in course notes); Wilke-Chang φ = 1.0, V_A(propane) = 74.5 cm³/mol."""
    from packages.dwsim_client.tools_mass_transfer import diffusivity_estimate, packed_column_height
    fid = T.new_flowsheet("packed", COMPS, "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=100.0, mole_fractions=GAS)
    T.add_material_stream(fid, "oil", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=120.0, mole_fractions=OIL)
    T.add_material_stream(fid, "top"); T.add_material_stream(fid, "bot")
    T.add_unit(fid, "absorption_column", "ABS", {"stages": 6, "top_P_Pa": 10e5})
    T.connect(fid, "ABS", feeds=["oil", "gas"], products=["top", "bot"])
    assert T.solve(fid)["solved"]
    K = k_propane(120.0)
    DG = diffusivity_estimate("fuller", 300.0, 10e5, 44.097, 16.043, 3 * 16.5 + 8 * 1.98, 16.5 + 4 * 1.98)["D_m2_s"]
    from packages.dwsim_client.tools_transport import _phase_props
    p = _phase_props(fid, "oil")
    DL = diffusivity_estimate("wilke_chang", 300.0, mu_solvent_cP=p["viscosity_Pa_s"] * 1e3, M_solvent=142.28, phi_solvent=1.0,
                              V_solute_cm3_mol=74.5)["D_m2_s"]
    r = packed_column_height(fid, "gas", "oil", 1.0, 105.0, 0.050, 0.075, K, 0.02, 0.94, DL, DG)
    T.close_flowsheet(fid)
    return {"K": K, "D_G_m2_s": DG, "D_L_m2_s": DL, **{k: r[k] for k in ("absorption_factor_A", "N_OG", "H_G_m", "H_L_m", "H_OG_m", "height_m",
                                                                    "stages_equivalent_kremser", "HETP_m", "gas_film_resistance_fraction")},
            "a_w_m2_m3": r["onda"]["a_w_m2_m3"], "kL_m_s": r["onda"]["kL_m_s"], "kG_m_s": r["onda"]["kG_m_s"]}


def part_a():
    for L in (30.0, 60.0, 120.0):
        K = k_propane(L)
        for N in (4, 6, 10):
            c = column(L, N)
            A = 0.5 * (L + c["L_bot"]) / (K * 0.5 * (100.0 + c["V_top"]))
            fk = kremser_fraction_absorbed(A, N)
            yield {"L_mol_s": L, "N": N, "K_propane": K, "A": A, "frac_kremser": fk, **c,
                   "diff_pts": (fk - c["frac_dwsim"]) * 100}



if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "A") == "A":
        for row in part_a():
            print(json.dumps(row), flush=True)
    else:
        print(json.dumps(part_b()), flush=True)
