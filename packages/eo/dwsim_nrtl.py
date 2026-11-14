"""Independent check of the NRTL EO column in DWSIM (bloque M15): the same column rebuilt in DWSIM's rigorous solver
with its NRTL property package and THE SAME binary parameters, written into the package from Python.

DWSIM (GPL-3) runs through pythonnet in a separate process from IDAES's model; nothing of DWSIM is copied here.
DWSIM's NRTL (DWSIM.Thermodynamics.PropertyPackages.NRTLPropertyPackage, m_uni.InteractionParameters[i][j]) uses
  tau_ij = (A_ij + B_ij·T + C_ij·T²) / (R·T),  R = 1.98721 cal/(mol·K),  alpha_ij = alpha12
so tau = a + b/T maps to A_ij = b_ij·R_cal (cal/mol), B_ij = a_ij·R_cal, C_ij = 0. The ChemSep NRTL table in thermo is
the same one DWSIM loads (ethanol/water A12 = −57.9601 cal/mol ↔ b12 = −29.1667 K), so the mapping is checked by
`read_parameters` on every call.
"""
from __future__ import annotations

R_CAL = 1.98721            # cal/(mol·K), the constant DWSIM's NRTL divides by (ChemSep convention)
ENTHALPY_MODES = ("LeeKesler", "Ideal", "Excess", "ExpData", "LiqCp_Excess")


def _pp(fid: str):
    from packages.dwsim_client import tools as T
    fs = T.SESSIONS[fid]["fs"]
    pp = list(fs.Inner.PropertyPackages.Values)[0]
    return getattr(pp, "__implementation__", pp)


def read_parameters(fid: str, dwsim_names: list[str]) -> dict:
    """DWSIM's stored NRTL parameters for every pair, converted to (a_ij, b_ij K, alpha)."""
    ip = _pp(fid).m_uni.InteractionParameters
    out = {}
    for i, a in enumerate(dwsim_names):
        for b in dwsim_names[i + 1:]:
            for x, y, flip in ((a, b, False), (b, a, True)):
                if ip.ContainsKey(x) and ip[x].ContainsKey(y):
                    o = ip[x][y]
                    A12, A21, B12, B21 = (float(getattr(o, k)) for k in ("A12", "A21", "B12", "B21"))
                    rec = {"b_ij_K": A12 / R_CAL, "b_ji_K": A21 / R_CAL, "a_ij": B12 / R_CAL, "a_ji": B21 / R_CAL,
                           "alpha": float(o.alpha12), "C12": float(o.C12), "C21": float(o.C21),
                           "stored_as": f"{x}/{y}", "A12_cal_mol": A12, "A21_cal_mol": A21}
                    if flip:
                        rec["b_ij_K"], rec["b_ji_K"] = rec["b_ji_K"], rec["b_ij_K"]
                        rec["a_ij"], rec["a_ji"] = rec["a_ji"], rec["a_ij"]
                    out[f"{a}/{b}"] = rec
                    break
    return out


def set_parameters(fid: str, params: dict, dwsim_names: dict[str, str]) -> dict:
    """Write params (packages.eo.nrtl_properties.resolve_parameters) into DWSIM's NRTL package. dwsim_names maps
    each EO component name to its DWSIM compound name. Returns what DWSIM holds afterwards."""
    from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import NRTL_IPData
    from System import String
    from System.Collections.Generic import Dictionary
    pp = _pp(fid)
    ip = pp.m_uni.InteractionParameters
    names = params["names"]
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            da, db = dwsim_names[a], dwsim_names[b]
            for x, y in ((db, da),):          # remove a reversed entry so only one record defines the pair
                if ip.ContainsKey(x) and ip[x].ContainsKey(y):
                    ip[x].Remove(y)
            if not ip.ContainsKey(da):
                ip.Add(da, Dictionary[String, NRTL_IPData]())
            if not ip[da].ContainsKey(db):
                ip[da].Add(db, NRTL_IPData())
            o = ip[da][db]
            o.A12 = params["tau_b_K"][i][j] * R_CAL
            o.A21 = params["tau_b_K"][j][i] * R_CAL
            o.B12 = params["tau_a"][i][j] * R_CAL
            o.B21 = params["tau_a"][j][i] * R_CAL
            o.C12 = 0.0
            o.C21 = 0.0
            o.alpha12 = params["alpha"][i][j]
    return read_parameters(fid, [dwsim_names[n] for n in names])


def configure(fid: str, poynting: bool | None = None, enthalpy_mode: str | None = None) -> dict:
    import System
    pp = _pp(fid)
    if poynting is not None:
        pp.LiquidFugacity_UsePoyntingCorrectionFactor = bool(poynting)
    if enthalpy_mode is not None:
        if enthalpy_mode not in ENTHALPY_MODES:
            raise ValueError(f"enthalpy_mode en {ENTHALPY_MODES}")
        t = pp.EnthalpyEntropyCpCvCalculationMode.GetType()
        pp.EnthalpyEntropyCpCvCalculationMode = System.Enum.Parse(t, enthalpy_mode)
    return {"poynting": bool(pp.LiquidFugacity_UsePoyntingCorrectionFactor),
            "enthalpy_mode": str(pp.EnthalpyEntropyCpCvCalculationMode),
            "vapor_fugacity": str(pp.VaporPhaseFugacityCalculationMode)}


def column(compounds_dwsim: list[str], feed_T_K: float, feed_P_Pa: float, flows_mol_s: dict, stages: int,
           feed_stage: int, reflux_ratio: float, pressure_Pa: float, bottoms_mol_s: float,
           params: dict | None = None, dwsim_names: dict | None = None,
           poynting: bool | None = False, enthalpy_mode: str | None = None, name: str = "eo_nrtl") -> dict:
    """DWSIM rigorous column (condenser + trays + reboiler = stages), NRTL package, specified by reflux ratio and
    bottoms molar flow (the client's distillate-flow spec replaces the reflux spec, both being condenser specs).
    params=None keeps DWSIM's own databank parameters (reported)."""
    from packages.dwsim_client import tools as T
    fid = T.new_flowsheet(name, compounds_dwsim, "NRTL")["flowsheet_id"]
    try:
        before = read_parameters(fid, compounds_dwsim)
        held = set_parameters(fid, params, dwsim_names) if params is not None else before
        settings = configure(fid, poynting, enthalpy_mode)
        T.add_material_stream(fid, "feed", temperature_K=feed_T_K, pressure_Pa=feed_P_Pa,
                              molar_flow_mol_s=sum(flows_mol_s.values()), compound_molar_flows_mol_s=flows_mol_s)
        for n in ("dist", "bot"):
            T.add_material_stream(fid, n)
        for n in ("cd", "rd"):
            T.add_energy_stream(fid, n)
        spec = {"stages": stages, "feed_stage": feed_stage, "reflux_ratio": reflux_ratio, "top_P_Pa": pressure_Pa,
                "bottoms_molar_flow_mol_s": bottoms_mol_s}
        T.add_unit(fid, "distillation_column", "T-1", spec)
        T.connect(fid, "T-1", feeds=["feed"], products=["dist", "bot"], energy_products=["cd"], energy_feeds=["rd"])
        sol = T.solve(fid)
        out = {"ok": sol["ok"], "solver_errors": sol["solver_errors"], "parameters_before": before,
               "parameters_used": held, "settings": settings}
        if sol["ok"]:
            u = T.read_unit(fid, "T-1")
            out.update(distillate=T.read_stream(fid, "dist"), bottoms=T.read_stream(fid, "bot"),
                       reboiler_duty_kW=u["energy_streams_kW"]["rd"], condenser_duty_kW=u["energy_streams_kW"]["cd"])
        return out
    finally:
        T.close_flowsheet(fid)
