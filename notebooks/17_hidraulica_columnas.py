"""Notebook 17 — packed-column hydraulics (bloque 23) and the first fully sized absorber.

A) Stichlmair et al. (1989) worked example through `fluids` (wet ΔP and flooding).
B) The propane absorber of notebook 16 (100 mol/s gas at 10 bar, 120 mol/s n-decane): diameter at 70 % of flooding for every
   packing in the catalog, wet ΔP per metre.
C) Pall rings 25 mm end to end: diameter (B) → packed height for 94 % recovery (Onda + Colburn, illustrative σc and
   diffusivities, see MOCKS.md) → total ΔP → shell purchase cost (Seider via IDAES, carbon steel, packed height + 3 m).
Usage: PYTHONPATH=. uv run python notebooks/17_hidraulica_columnas.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from packages.dwsim_client import tools as T  # noqa: E402
from packages.dwsim_client.tools_mass_transfer import diffusivity_estimate, packed_column_height, packed_column_hydraulics  # noqa: E402
from packages.dwsim_client.tools_transport import _phase_props  # noqa: E402
from packages.transport.column_hydraulics import PACKINGS, flooding_velocity, wet_dp_per_m  # noqa: E402

spec = importlib.util.spec_from_file_location("nb16", Path(__file__).with_name("16_transferencia_de_masa.py"))
nb16 = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb16)


def absorber_flowsheet():
    fid = T.new_flowsheet("abs17", nb16.COMPS, "PengRobinson")["flowsheet_id"]
    T.add_material_stream(fid, "gas", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=100.0, mole_fractions=nb16.GAS)
    T.add_material_stream(fid, "oil", temperature_K=300.0, pressure_Pa=10e5, molar_flow_mol_s=120.0, mole_fractions=nb16.OIL)
    T.add_material_stream(fid, "top"); T.add_material_stream(fid, "bot")
    T.add_unit(fid, "absorption_column", "ABS", {"stages": 6, "top_P_Pa": 10e5})
    T.connect(fid, "ABS", feeds=["oil", "gas"], products=["top", "bot"])
    assert T.solve(fid)["solved"]
    return fid


if __name__ == "__main__":
    ex = dict(voidage=0.68, specific_area=260.0, C1=32.0, C2=7.0, C3=1.0)
    print(json.dumps({"part": "A", "flood_m_s": flooding_velocity(5e-3, 5.0, 1200.0, 5e-5, ex), "flood_paper": 0.6394323542746928,
                      "wet_Pa_m": wet_dp_per_m(0.4, 5e-3, 5.0, 1200.0, 5e-5, ex), "wet_paper": 539.876823725352}))
    fid = absorber_flowsheet()
    for pk in PACKINGS:
        r = packed_column_hydraulics(fid, "gas", "oil", packing=pk)
        print(json.dumps({"part": "B", "packing": pk, **{k: r[k] for k in ("diameter_m", "gas_velocity_m_s", "liquid_velocity_m_s",
                                                                       "flooding_velocity_m_s", "F_factor_Pa05", "dP_wet_Pa_per_m")}}))
    hyd = packed_column_hydraulics(fid, "gas", "oil", packing="Pall rings 25 mm")
    K = nb16.k_propane(120.0)
    DG = diffusivity_estimate("fuller", 300.0, 10e5, 44.097, 16.043, 3 * 16.5 + 8 * 1.98, 16.5 + 4 * 1.98)["D_m2_s"]
    DL = diffusivity_estimate("wilke_chang", 300.0, mu_solvent_cP=_phase_props(fid, "oil")["viscosity_Pa_s"] * 1e3, M_solvent=142.28,
                              phi_solvent=1.0, V_solute_cm3_mol=74.5)["D_m2_s"]
    h = packed_column_height(fid, "gas", "oil", hyd["diameter_m"], 205.0, 0.025, 0.075, K, 0.02, 0.94, DL, DG)
    from packages.costing.sslw import cost_vessel
    length = h["height_m"] + 3.0
    c = cost_vessel(hyd["diameter_m"], length, vertical=True, material="CarbonSteel")
    print(json.dumps({"part": "C", "packing": "Pall rings 25 mm", "diameter_m": hyd["diameter_m"], "packed_height_m": h["height_m"],
                      "H_OG_m": h["H_OG_m"], "N_OG": h["N_OG"], "HETP_m": h["HETP_m"], "dP_total_kPa": hyd["dP_wet_Pa_per_m"] * h["height_m"] / 1e3,
                      "shell_length_m": length, "shell_purchase_usd_2023": c["purchase_cost_usd"], "wetted_fraction": h["onda"]["wetted_fraction"]}))
    T.close_flowsheet(fid)
