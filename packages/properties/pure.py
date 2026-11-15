"""Pure-compound property lookup with provenance (bloque `props`).

What it answers: «what is the Tc / Psat / ΔHf / viscosity of this compound, and how far can I trust the number?» without
building a flowsheet. Nothing is typed here: every value comes from the databases and correlations that ship with
`chemicals` and `thermo` (MIT, ChEDL), and each one is returned with

  * `method`  the library's method name (HEOS, CRC, DIPPR_PERRY_8E, JOBACK...), i.e. the data source or the estimator;
  * `basis`   "data" (measured value, evaluated compilation or a correlation fitted to data / to a reference equation of
              state), "compilation" (handbook that mixes measured and estimated numbers without telling them apart, e.g.
              Yaws, or an unreviewed one, e.g. Wikidata), "derived" (computed from other tabulated values), "estimate"
              (group contribution or corresponding states) or "unclassified";
  * for temperature-dependent properties the validity range of the method and whether T is inside it (`in_range`);
    outside every method's range the default correlation is extrapolated and the value says so (`extrapolated`).

Choice of method for a temperature-dependent property (the only decision taken here, reported in each value):
  1. the library's default method if T is inside its range; otherwise
  2. the best-ranked method with basis "data" whose range contains T; otherwise
  3. the best-ranked method of any basis whose range contains T (an estimate, flagged); otherwise
  4. the library's default method extrapolated (flagged).

Transport properties (viscosity, thermal conductivity, surface tension; block L4): measured on 2026-11-15 against NIST
WebBook in 19 fluids (tests/test_transport_properties.py), CoolProp's dedicated reference correlation of a fluid replaces
thermo's value only where it matched NIST within 1 % (several of CoolProp's correlations are older than REFPROP's and
deviate: n-pentane viscosity up to 45 %, ammonia 13 %); elsewhere thermo's value is kept, CoolProp's correlation is
attached as `reference_check` and disagreements are flagged. With P_Pa in the gas phase the dense-gas values come from
the reference correlation (dilute + residual + critical terms) when one exists.

Evidence: thermodynamic cross-checks between independent pieces of the library's data for this compound, not a check of
the user's data: Psat(Tb) against 1 atm, the acentric factor against its definition, and (if T is given) ΔHvap against
the Clapeyron equation with the saturated volumes."""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from contextlib import contextmanager
from functools import lru_cache, wraps

R_GAS = 8.314462618            # J/(mol K), CODATA 2018
P_ATM = 101325.0               # Pa
T_STD = 298.15                 # K, reference of the formation properties
PSAT_TB_TOL = 0.02             # |Psat(Tb)/1 atm - 1| accepted between the Tb source and the vapour-pressure correlation
OMEGA_TOL = 0.02               # |omega(definition from Psat) - omega(tabulated)|
CLAPEYRON_TOL = 0.03           # |ΔHvap(correlation)/ΔHvap(Clapeyron) - 1|, only counted with data-based saturated volumes
SAT_MARGIN = 1e-3              # the saturated vapour volume is evaluated at Psat·(1 - margin) to stay on the gas side
CLAPEYRON_MAX_TR = 0.95        # above this reduced temperature the Clapeyron check is ill-conditioned (Vg - Vl -> 0)
PSAT_TC_TOL = 0.03             # |Psat(Tc)/Pc - 1|: the vapour-pressure correlation must end at the tabulated critical point
SPREAD_WARN = 0.05             # relative spread among data-based methods that triggers a warning
# transport properties (block L4, tests/test_transport_properties.py)
NIST_AGREE_TOL = 0.01          # a CoolProp transport correlation replaces thermo's value only if it matched NIST within this
TRANSPORT_CHECK_TOL = 0.02     # |thermo value / CoolProp reference correlation - 1| above which the value is flagged
METHOD_BAND_FLAG = 0.05        # p90 of a thermo method's measured deviation above which its values are flagged
NEAR_CRITICAL_TR = 0.95        # above this T/Tc the liquid transport properties of independent sources diverge (measured)
DILUTE_RHO_MOL_M3 = 1e-6       # molar density at which CoolProp is evaluated for the dilute-gas (zero-density) limit
DENSE_GAS_RHO_R = 0.02         # ρ/ρc above which a dilute-gas transport value is warned (N2 at ρ/ρc 0.18: -2 to -5 %)
# library methods that are data but NOT the quantity reported here (excluded from the spread among data methods)
_OTHER_QUANTITY = {("Cp_ideal_gas_J_molK", "VDI_TABULAR"): "Cp del vapor saturado real, no del gas ideal"}

# ---- provenance of the library's methods -----------------------------------------------------------------------------
_CONST_DATA = {"HEOS", "IUPAC", "MATTHEWS", "CRC", "CRC_ORG", "CRC_INORG", "PD", "WEBBOOK", "PSRK", "PINAMARTINES",
               "COMMON_CHEMISTRY", "OPEN_NTBKM", "ATCT_G", "ATCT_L", "API_TDB_G", "TRC", "JANAF", "STAVELEY", "CCCBDB",
               "MULLER", "POLING"}
_CONST_COMPILATION = {"YAWS", "WIKIDATA"}
_CONST_DERIVED = {"ACENTRIC_DEFINITION", "MELTING", "PSI4_2022A"}
_CONST_ESTIMATE = {"JOBACK", "WILSON_JASPERSON", "FEDORS", "GRIGORAS", "IHMELS", "MEISSNER", "LK"}
_CONST_NOTES = {
    "HEOS": "constante de la ecuación de estado de referencia (Helmholtz) que usa CoolProp",
    "IUPAC": "valor crítico evaluado por IUPAC (Ambrose, Tsonopoulos, Gude, Teja y otros)",
    "CRC": "CRC Handbook of Chemistry and Physics", "CRC_ORG": "CRC Handbook (compuestos orgánicos)",
    "CRC_INORG": "CRC Handbook (compuestos inorgánicos)", "WEBBOOK": "NIST Chemistry WebBook (copia incluida en chemicals)",
    "PSRK": "tabla de constantes del modelo PSRK (Horstmann y otros, 2005)", "PD": "Passut y Danner (1973)",
    "MATTHEWS": "Mathews (1972), constantes críticas de inorgánicos", "PINAMARTINES": "Piña-Martinez y otros (2021)",
    "COMMON_CHEMISTRY": "CAS Common Chemistry", "OPEN_NTBKM": "Open Notebook Melting Point (Bradley y otros)",
    "ATCT_G": "Active Thermochemical Tables (gas)", "ATCT_L": "Active Thermochemical Tables (líquido)",
    "API_TDB_G": "API Technical Data Book (gas)", "TRC": "tablas TRC", "JANAF": "tablas NIST-JANAF",
    "STAVELEY": "Staveley y otros (1981), puntos triples", "CCCBDB": "NIST CCCBDB (dipolo experimental)",
    "MULLER": "Muller y otros (2012)", "POLING": "Poling, Prausnitz y O'Connell (2001)",
    "YAWS": "compilación de Yaws: mezcla valores medidos y estimados sin distinguirlos",
    "WIKIDATA": "Wikidata: compilación abierta sin revisión",
    "ACENTRIC_DEFINITION": "calculado de la definición ω = −log10(Psat(0.7·Tc)/Pc) − 1 con la presión de vapor de la "
                           "librería",
    "MELTING": "aproximado por el punto de fusión normal (Tt ≈ Tm)",
    "PSI4_2022A": "cálculo cuántico (Psi4), no medido",
    "JOBACK": "estimación por contribución de grupos de Joback",
    "WILSON_JASPERSON": "estimación por contribución de grupos de Wilson-Jasperson",
    "FEDORS": "estimación por contribución de grupos de Fedors",
}

_TDEP_DATA = {
    "IAPWS_PSAT", "IAPWS_SIGMA", "HEOS_FIT", "REFPROP_FIT", "COOLPROP", "REFPROP", "WAGNER_MCGARRY", "WAGNER_POLING",
    "ANTOINE_EXTENDED_POLING", "ANTOINE_POLING", "ANTOINE_WEBBOOK", "DIPPR_PERRY_8E", "VDI_PPDS", "VDI_TABULAR", "LANDOLT",
    "ALCOCK_ELEMENTS", "CRC_HVAP_TB", "CRC_HVAP_298", "TRCIG", "WEBBOOK_SHOMATE", "POLING_POLY", "POLING_CONST", "CRCSTD",
    "JANAF", "UNARY", "DUTT_PRASAD", "VISWANATH_NATARAJAN_3", "VISWANATH_NATARAJAN_2", "VISWANATH_NATARAJAN_2E", "JASPER",
    "SOMAYAJULU", "SOMAYAJULU2", "CRC_INORG_L", "CRC_INORG_L_CONST", "COMMON_CHEMISTRY", "CRC_VIRIAL"}
_TDEP_ESTIMATE = {
    "AMBROSE_WALTON", "LEE_KESLER_PSAT", "EDALAT", "EOS", "BOILING_CRITICAL", "SANJARI", "MORGAN_KOBAYASHI",
    "SIVARAMAN_MAGEE_KOBAYASHI", "VELASCO", "PITZER", "ALIBAKHSHI", "GHARAGHEIZI_HVAP_298", "CLAPEYRON", "RIEDEL", "CHEN",
    "VETERE", "LIU", "LASTOVKA_SHAW", "JOBACK", "ROWLINSON_POLING", "ROWLINSON_BONDI", "DADGOSTAR_SHAW", "MMSNM0",
    "HTCOSTALDFIT", "RACKETTFIT", "MMSNM0FIT",
    "HTCOSTALD", "YEN_WOODS_SAT", "RACKETT", "YAMADA_GUNN", "BHIRUD_NORMAL", "TOWNSEND_HALES", "CAMPBELL_THODOS",
    "TSONOPOULOS_EXTENDED", "TSONOPOULOS", "ABBOTT", "PITZER_CURL", "IDEAL", "LETSOU_STIEL", "PRZEDZIECKI_SRIDHAR",
    "GHARAGHEIZI", "YOON_THODOS", "STIEL_THODOS", "LUCAS_GAS", "GHARAGHEIZI_L", "SHEFFY_JOHNSON", "SATO_RIEDEL",
    "LAKSHMI_PRASAD", "BAHADORI_L", "NICOLA", "NICOLA_ORIGINAL", "GHARAGHEIZI_G", "DIPPR_9B", "CHUNG", "ELI_HANLEY",
    "EUCKEN_MOD", "EUCKEN", "BAHADORI_G", "MIQUEU", "BROCK_BIRD", "SASTRI_RAO", "PITZER_SIGMA", "ZUO_STENBY", "Aleem"}
_TDEP_NOTES = {
    "IAPWS_PSAT": "IAPWS-95 (curva de saturación del agua)",
    "IAPWS_SIGMA": "correlación IAPWS de tensión superficial del agua",
    "HEOS_FIT": "ajuste de thermo a la ecuación de estado de referencia (Helmholtz) de CoolProp",
    "REFPROP_FIT": "ajuste de thermo a REFPROP (NIST)", "COOLPROP": "ecuación de estado de referencia vía CoolProp",
    "REFPROP": "correlación de tensión superficial de REFPROP (Mulero y otros)",
    "WAGNER_MCGARRY": "ecuación de Wagner con coeficientes de McGarry (1983)",
    "WAGNER_POLING": "ecuación de Wagner con coeficientes de Poling y otros (2001)",
    "ANTOINE_EXTENDED_POLING": "Antoine extendida de Poling y otros (2001)",
    "ANTOINE_POLING": "Antoine de Poling y otros (2001)",
    "ANTOINE_WEBBOOK": "Antoine del NIST Chemistry WebBook",
    "DIPPR_PERRY_8E": "correlación DIPPR tabulada en Perry's 8.ª ed.", "VDI_PPDS": "correlación PPDS del VDI Heat Atlas",
    "VDI_TABULAR": "interpolación de la tabla de saturación del VDI Heat Atlas", "LANDOLT": "Landolt-Börnstein",
    "TRCIG": "ecuación TRC de gas ideal (Kabo y otros, 2002)", "WEBBOOK_SHOMATE": "ecuación de Shomate del NIST WebBook",
    "POLING_POLY": "polinomio de gas ideal de Poling y otros (2001)", "POLING_CONST": "valor constante de Poling a 298 K",
    "CRCSTD": "valor constante del CRC a 298 K", "CRC_HVAP_TB": "ΔHvap del CRC en Tb escalado con Watson",
    "CRC_HVAP_298": "ΔHvap del CRC a 298 K escalado con Watson", "JASPER": "Jasper (1972)",
    "SOMAYAJULU": "Somayajulu (1988)", "SOMAYAJULU2": "Somayajulu (1988), revisada",
    "CRC_VIRIAL": "coeficientes viriales del CRC",
    "CRC_INORG_L": "densidad de inorgánicos fundidos del CRC",
    "CRC_INORG_L_CONST": "densidad constante del CRC (un solo punto)",
    "HTCOSTALDFIT": "COSTALD (estados correspondientes) con el volumen característico ajustado a datos",
    "RACKETTFIT": "Rackett (estados correspondientes) con Z_RA ajustado a datos",
    "MMSNM0FIT": "SNM0 (estados correspondientes) con un parámetro ajustado a datos", "DUTT_PRASAD": "Dutt y Prasad",
    "UNARY": "sistema unario (elemento)",
    "VISWANATH_NATARAJAN_3": "Viswanath y Natarajan (3 parámetros)", "VISWANATH_NATARAJAN_2": "Viswanath y Natarajan",
    "VISWANATH_NATARAJAN_2E": "Viswanath y Natarajan", "ALCOCK_ELEMENTS": "Alcock y otros (elementos)",
    "COMMON_CHEMISTRY": "CAS Common Chemistry (un solo punto)", "JANAF": "tablas NIST-JANAF",
    "IDEAL": "gas ideal", "EOS": "ecuación cúbica de estado con Tc, Pc y ω (predicción)",
    "TSONOPOULOS_EXTENDED": "virial truncado en B con la correlación de Tsonopoulos extendida (estados correspondientes)",
}
_BASIS_FLAG = {
    "compilation": "valor de compilación sin distinguir medido de estimado",
    "derived": "valor derivado de otros, no medido directamente",
    "estimate": "VALOR ESTIMADO (contribución de grupos o estados correspondientes), no medido",
    "unclassified": "origen del método no clasificado por esta herramienta: tratar como no verificado",
}

# Spanish names of everyday compounds that the library's index (mostly English) does not carry
_ES_ALIASES = {
    "agua": "water", "amoniaco": "ammonia", "benceno": "benzene", "tolueno": "toluene", "acetona": "acetone",
    "acido acetico": "acetic acid", "hexano": "hexane", "n-hexano": "hexane", "metano": "methane", "etano": "ethane",
    "propano": "propane", "butano": "butane", "n-butano": "butane", "pentano": "pentane", "heptano": "heptane",
    "octano": "octane", "dioxido de carbono": "carbon dioxide", "monoxido de carbono": "carbon monoxide",
    "nitrogeno": "nitrogen", "oxigeno": "oxygen", "hidrogeno": "hydrogen", "metanol": "methanol", "etanol": "ethanol",
    "butanol": "1-butanol", "n-butanol": "1-butanol", "1-butanol": "1-butanol", "propanol": "1-propanol",
    "isopropanol": "isopropanol", "acido sulfurico": "sulfuric acid", "acido clorhidrico": "hydrogen chloride",
    "cloro": "chlorine", "etileno": "ethylene", "propileno": "propylene", "estireno": "styrene", "fenol": "phenol",
    "glicerina": "glycerol", "glicerol": "glycerol", "etilenglicol": "ethylene glycol", "ciclohexano": "cyclohexane",
    "xileno": "xylene", "sulfuro de hidrogeno": "hydrogen sulfide", "acido sulfhidrico": "hydrogen sulfide",
    "dioxido de azufre": "sulfur dioxide", "cloroformo": "chloroform", "eter dietilico": "diethyl ether", "argon": "argon",
    "helio": "helium", "aire": "air"}


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.strip().lower()) if not unicodedata.combining(c))


def _basis_const(method: str) -> str:
    if method in _CONST_DATA:
        return "data"
    if method in _CONST_COMPILATION:
        return "compilation"
    if method in _CONST_DERIVED:
        return "derived"
    if method in _CONST_ESTIMATE:
        return "estimate"
    return "unclassified"


def _basis_tdep(method: str) -> str:
    root = re.sub(r"_(\d+)$", "", method)
    if method in _TDEP_DATA or root in _TDEP_DATA or method.startswith("ZABRANSKY"):
        return "data"
    if method in _TDEP_ESTIMATE:
        return "estimate"
    return "unclassified"


def _note_tdep(method: str) -> str:
    if method.startswith("ZABRANSKY"):
        return "correlación de Zábranský y otros (1996) para Cp de líquidos"
    if method.startswith("Fit 20"):
        return "ajuste propio de thermo cuyo origen de datos no está documentado en la librería"
    return _TDEP_NOTES.get(method, "estimación por estados correspondientes o contribución de grupos"
                           if _basis_tdep(method) == "estimate" else "método de thermo")


# ---- identification ---------------------------------------------------------------------------------------------------
def suggest_names(query: str, n: int = 5) -> list[str]:
    """Close names from the library's own name index (difflib); empty if none is close."""
    from chemicals import identifiers
    db = identifiers.get_pubchem_db()
    try:
        db.autoload_main_db()
    except Exception:  # noqa: BLE001 — the index may already be loaded
        pass
    q = _fold(query)
    pool = [k for k in db.name_index if abs(len(k) - len(q)) <= 3]
    close = difflib.get_close_matches(q, pool, n=n, cutoff=0.78)
    close += [k for k in difflib.get_close_matches(q, list(_ES_ALIASES), n=2, cutoff=0.78) if k not in close]
    return close[:n]


def identify(compound: str):
    """`chemicals` metadata of a compound given by name, CAS number, formula, SMILES or InChI. Spanish error if unknown."""
    from chemicals.identifiers import search_chemical
    if not isinstance(compound, str) or not compound.strip():
        raise ValueError("Falta 'compound': el nombre (en inglés o español común), número CAS, fórmula o SMILES del "
                         "compuesto cuyas propiedades se consultan.")
    q = compound.strip()
    alias = _ES_ALIASES.get(_fold(q))
    for attempt in ([alias] if alias else []) + [q]:
        try:
            return search_chemical(attempt), ("alias en español → " + alias if attempt == alias and alias != q else None)
        except Exception:  # noqa: BLE001 — chemicals raises ValueError for unknown identifiers
            continue
    hints = suggest_names(q)
    extra = f" Nombres parecidos en la base de datos: {', '.join(hints)}." if hints else \
        " La base de datos no tiene nombres parecidos; prueba con el número CAS o el nombre en inglés."
    raise ValueError(f"Compuesto no reconocido por la base de datos de chemicals: «{compound}».{extra}")


def _identified_by(query: str, meta) -> str:
    q = query.strip()
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", q):
        return "CAS"
    if q.replace(" ", "") == meta.formula and q.lower() not in (meta.common_name or "").lower():
        return "formula"
    if q in (meta.smiles, meta.InChI, meta.InChI_key) or q.lower().startswith(("inchi=", "smiles=")):
        return "structure"
    return "name"


# ---- constants --------------------------------------------------------------------------------------------------------
def _constant(cas: str, func, methods_func, scale: float = 1.0) -> dict:
    try:
        methods = list(methods_func(cas))
    except Exception:  # noqa: BLE001
        methods = []
    vals = []
    for m in methods:
        try:
            v = func(cas, method=m)
        except Exception:  # noqa: BLE001
            v = None
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            vals.append((m, float(v) * scale))
    if not vals:
        return {"value": None, "method": None, "basis": None, "estimated": None,
                "note": "la base de datos de chemicals no tiene este dato para el compuesto"}
    m0, v0 = vals[0]
    b0 = _basis_const(m0)
    out = {"value": v0, "method": m0, "basis": b0, "estimated": b0 == "estimate",
           "note": _CONST_NOTES.get(m0, "fuente de chemicals")}
    if b0 != "data":
        out["flag"] = _BASIS_FLAG[b0]
    out["alternatives"] = [{"method": m, "value": v, "basis": _basis_const(m)} for m, v in vals[1:]]
    data_vals = [v for m, v in vals if _basis_const(m) == "data"]
    if len(data_vals) >= 2:
        out["n_data_sources"] = len(data_vals)
        out["data_sources_spread"] = max(data_vals) - min(data_vals)
    return out


def _derived(value, note: str, sources: list[str]) -> dict:
    if value is None or not math.isfinite(value):
        return {"value": None, "method": None, "basis": None, "estimated": None, "note": "no se puede derivar: " + note}
    return {"value": float(value), "method": "DERIVED", "basis": "derived", "estimated": False, "note": note,
            "derived_from": sources, "flag": _BASIS_FLAG["derived"]}


def _constants(cas: str, pkg) -> dict:
    from chemicals import acentric, critical, dipole, phase_change, reaction, triple
    c = {
        "Tc_K": _constant(cas, critical.Tc, critical.Tc_methods),
        "Pc_Pa": _constant(cas, critical.Pc, critical.Pc_methods),
        "Vc_m3_mol": _constant(cas, critical.Vc, critical.Vc_methods),
        "acentric_factor": _constant(cas, acentric.omega, acentric.omega_methods),
        "Tb_K": _constant(cas, phase_change.Tb, phase_change.Tb_methods),
        "Tm_K": _constant(cas, phase_change.Tm, phase_change.Tm_methods),
        "Tt_K": _constant(cas, triple.Tt, triple.Tt_methods),
        "Pt_Pa": _constant(cas, triple.Pt, triple.Pt_methods),
        "Hfus_J_mol": _constant(cas, phase_change.Hfus, phase_change.Hfus_methods),
        "Hf_gas_J_mol": _constant(cas, reaction.Hfg, reaction.Hfg_methods),
        "Hf_liquid_J_mol": _constant(cas, reaction.Hfl, reaction.Hfl_methods),
        "Hf_solid_J_mol": _constant(cas, reaction.Hfs, reaction.Hfs_methods),
        "S0_gas_J_molK": _constant(cas, reaction.S0g, reaction.S0g_methods),
        "dipole_moment_debye": _constant(cas, dipole.dipole_moment, dipole.dipole_moment_methods),
    }
    tc, pc, vc = (c[k]["value"] for k in ("Tc_K", "Pc_Pa", "Vc_m3_mol"))
    c["Zc"] = _derived(pc * vc / (R_GAS * tc) if None not in (tc, pc, vc) else None, "Zc = Pc·Vc/(R·Tc)",
                       [f"Tc {c['Tc_K']['method']}", f"Pc {c['Pc_Pa']['method']}", f"Vc {c['Vc_m3_mol']['method']}"])
    if "estimate" in (c["Tc_K"]["basis"], c["Pc_Pa"]["basis"], c["Vc_m3_mol"]["basis"]) and c["Zc"]["value"] is not None:
        c["Zc"].update(estimated=True, flag="derivado de constantes críticas ESTIMADAS")
    # ΔfG°(gas, 298.15 K) = ΔfH° − T·ΔfS°, with the absolute entropies of the elements of chemicals.periodic_table (thermo)
    gf, sf = pkg.Gfgs[0], pkg.Sfgs[0]
    hf, s0 = c["Hf_gas_J_mol"], c["S0_gas_J_molK"]
    consistent = hf["value"] is not None and pkg.Hfgs[0] is not None \
        and abs(pkg.Hfgs[0] - hf["value"]) < 1e-6 * max(1.0, abs(hf["value"]))
    c["Gf_gas_J_mol"] = _derived(gf if consistent else None,
                                 "ΔfG°gas(298.15 K) = ΔfH°gas − T·(S°gas − ΣS° de los elementos); entropías de los "
                                 "elementos de chemicals.periodic_table, cálculo de thermo.ChemicalConstantsPackage",
                                 [f"ΔfH°gas {hf['method']}", f"S°gas {s0['method']}"])
    if c["Gf_gas_J_mol"]["value"] is not None:
        c["Gf_gas_J_mol"]["Sf_gas_J_molK"] = float(sf) if sf is not None else None
        if "estimate" in (hf["basis"], s0["basis"]) or "compilation" in (hf["basis"], s0["basis"]):
            c["Gf_gas_J_mol"].update(estimated="estimate" in (hf["basis"], s0["basis"]),
                                     flag="derivado de ΔfH° o S° que no son datos medidos (ver sus campos)")
    return c


# ---- temperature-dependent properties ---------------------------------------------------------------------------------
def _calc(obj, T: float, method: str):
    try:
        v = obj.calculate(T, method)
    except Exception:  # noqa: BLE001 — a method may fail at T; it is then simply not available
        return None
    return float(v) if v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None


def _tdep(obj, T: float, alternatives: bool, unavailable: str | None = None, key: str = "") -> dict:
    """Value of a thermo TDependentProperty at T with the method that produced it, its range and its basis."""
    if unavailable:
        return {"value": None, "method": None, "note": unavailable}
    default = obj.method
    limits = dict(obj.T_limits)
    if default is None or default not in limits:
        return {"value": None, "method": None, "note": "thermo no tiene ningún método para esta propiedad de este compuesto"}
    ranked = [m for m in getattr(obj, "ranked_methods", []) if m in obj.all_methods and m in limits]
    ranked += [m for m in obj.all_methods if m not in ranked and m in limits]

    def inside(m: str) -> bool:
        return limits[m][0] <= T <= limits[m][1]

    candidates = [default] if inside(default) else []
    candidates += [m for m in ranked if inside(m) and _basis_tdep(m) == "data" and m not in candidates]
    candidates += [m for m in ranked if inside(m) and m not in candidates]
    chosen, value = None, None
    for m in candidates:
        value = _calc(obj, T, m)
        if value is not None:
            chosen = m
            break
    extrapolated = False
    if chosen is None:
        try:
            value = obj.T_dependent_property(T)
        except Exception:  # noqa: BLE001
            value = None
        if value is None or not math.isfinite(value) or value <= 0:
            return {"value": None, "method": default, "T_range_K": list(limits[default]), "in_range": False,
                    "note": f"T fuera del intervalo de todos los métodos y la extrapolación de {default} no da un valor "
                            "físico"}
        chosen, value, extrapolated = default, float(value), True
    basis = _basis_tdep(chosen)
    out = {"value": value, "method": chosen, "basis": basis, "estimated": basis == "estimate",
           "T_range_K": [float(limits[chosen][0]), float(limits[chosen][1])], "in_range": not extrapolated,
           "extrapolated": extrapolated, "note": _note_tdep(chosen)}
    flags = []
    if extrapolated:
        flags.append(f"VALOR EXTRAPOLADO: T = {T:g} K está fuera de {limits[chosen][0]:.1f}-{limits[chosen][1]:.1f} K "
                     f"(esquema de thermo: {obj.extrapolation})")
    if basis != "data":
        flags.append(_BASIS_FLAG[basis])
    if chosen != default and not extrapolated:
        out["library_default_method"] = default
        out["note"] += (f"; el método por defecto de thermo ({default}) no cubre T "
                        f"({limits[default][0]:.1f}-{limits[default][1]:.1f} K)")
    if flags:
        out["flag"] = "; ".join(flags)
    others = [(m, _calc(obj, T, m)) for m in ranked if m != chosen and inside(m)]
    others = [(m, v) for m, v in others if v is not None]
    data = [(m, v) for m, v in others if _basis_tdep(m) == "data" and (key, m) not in _OTHER_QUANTITY]
    data += [(chosen, value)] if basis == "data" and not extrapolated else []
    if len(data) >= 2:
        vals = sorted(v for _, v in data)
        median = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
        far = max(data, key=lambda mv: abs(mv[1] - median))
        out["n_data_methods"] = len(data)
        out["data_methods_spread_rel"] = (vals[-1] - vals[0]) / abs(median)
        out["data_methods_median"] = median
        out["value_vs_data_median_rel"] = value / median - 1.0
        out["data_methods_farthest"] = {"method": far[0], "value": far[1]}
    if alternatives:
        out["alternatives"] = [dict({"method": m, "value": v, "basis": _basis_tdep(m)},
                                    **({"note": _OTHER_QUANTITY[(key, m)]} if (key, m) in _OTHER_QUANTITY else {}))
                               for m, v in others]
    return out


def _gas_volume(vg, T: float, P: float) -> dict:
    """Molar volume of the gas at T, P: reference EOS (CoolProp) if the compound has one, else the truncated virial."""
    for m, basis in (("COOLPROP", "data"), ("TSONOPOULOS_EXTENDED", "estimate"), ("IDEAL", "estimate")):
        if m not in vg.all_methods_P:
            continue
        try:
            v = vg.calculate_P(T, P, m)
        except Exception:  # noqa: BLE001
            v = None
        if v is not None and math.isfinite(v) and v > 0:
            out = {"value": float(v), "method": m, "basis": basis, "estimated": basis == "estimate", "note": _note_tdep(m),
                   "Z": float(P * v / (R_GAS * T))}
            if m == "TSONOPOULOS_EXTENDED":
                out["flag"] = ("VALOR ESTIMADO: virial truncado en el segundo coeficiente; solo válido a densidad baja "
                               "(aprox. menos de la mitad de la densidad crítica)")
            elif m == "IDEAL":
                out["flag"] = ("VALOR ESTIMADO: gas ideal (el compuesto no tiene ecuación de referencia ni constantes "
                               "críticas)")
            return out
    return {"value": None, "method": None, "note": "thermo no tiene método de volumen de gas para este compuesto"}


# ---- transport properties: reference correlations through CoolProp ---------------------------------------------------
# max |CoolProp 8.0.0 / NIST WebBook - 1| measured on 2026-11-15 (tests/test_transport_properties.py, WebBook tables read
# that day) over the saturated liquid (6 T from near the triple point to T/Tc ≈ 0.99), the gas at 1 kPa and the gas at
# 5-10 MPa. Tuple: (viscosity, thermal conductivity); None = CoolProp has no correlation.
_CP_VS_NIST = {
    "109-66-0": (0.45, 0.16), "7664-41-7": (0.13, 0.14), "74-82-8": (0.038, 0.0075), "108-88-3": (0.00011, 0.002),
    "7727-37-9": (0.00026, 0.00022), "124-38-9": (0.00019, 0.00018), "74-98-6": (0.019, 0.0025),
    "67-56-1": (0.0001, 0.0001), "7783-06-4": (0.092, None), "110-54-3": (0.00013, 0.00091),
    "811-97-2": (0.00013, 0.00091), "1333-74-0": (0.0001, 0.00073), "7440-37-1": (0.00024, 0.00022),
    "74-84-0": (0.033, 0.015), "7782-44-7": (0.0003, 0.0001), "71-43-2": (0.0025, 0.0014), "106-97-8": (0.053, 0.0012),
    "142-82-5": (0.053, 0.0063), "7732-18-5": (0.0001, 0.0001),
}
# same for the surface tension (CoolProp: Mulero et al. 2012/2014): (T/Tc ≤ 0.95, T/Tc > 0.95)
_CP_SIGMA_VS_NIST = {
    "109-66-0": (0.00011, 0.00075), "7664-41-7": (0.0056, 0.068), "74-82-8": (0.00025, 0.0013),
    "108-88-3": (0.00013, 0.00086), "7727-37-9": (0.00025, 0.0024), "124-38-9": (0.0001, 0.002),
    "74-98-6": (0.00013, 0.00013), "67-56-1": (0.00014, 0.00051), "7783-06-4": (0.00014, 0.0014),
    "110-54-3": (0.00015, 0.0015), "811-97-2": (0.00012, 0.0011), "1333-74-0": (0.00016, 0.00025),
    "7440-37-1": (0.00023, 0.0022), "74-84-0": (0.00014, 0.0012), "7782-44-7": (0.00036, 0.0028),
    "71-43-2": (0.0001, 0.0005), "106-97-8": (0.0001, 0.0017), "142-82-5": (0.0022, 0.026), "7732-18-5": (0.011, 0.028),
}
# CAS keys of both tables (values rounded up to two significant figures):
# 109-66-0 n-Pentane, 7664-41-7 Ammonia, 74-82-8 Methane, 108-88-3 Toluene, 7727-37-9 Nitrogen, 124-38-9 CarbonDioxide,
# 74-98-6 Propane, 67-56-1 Methanol, 7783-06-4 HydrogenSulfide, 110-54-3 n-Hexane, 811-97-2 R134a, 1333-74-0 Hydrogen,
# 7440-37-1 Argon, 74-84-0 Ethane, 7782-44-7 Oxygen, 71-43-2 Benzene, 106-97-8 n-Butane, 142-82-5 n-Heptane,
# 7732-18-5 Water
# thermo methods kept in preference to CoolProp for one property (they are the ones NIST WebBook uses)
_THERMO_PREFERRED = {"sigma_N_m": {"IAPWS_SIGMA"}}
# measured |thermo method / reference correlation - 1| (p90, max, n points) on 2026-11-15: references = the CoolProp
# correlations that matched NIST WebBook within 1 % (mu: 11 fluids, k: 15, sigma: 18 at T/Tc ≤ 0.95); saturated liquid
# at 5 T from the triple point to 0.92 of the way to Tc, dilute gas at up to 6 T between 0.6·Tc and the EOS Tmax. thermo's
# own COOLPROP gas method is the gas at 1 atm or saturated vapour, not the dilute limit, hence its larger band.
_METHOD_BAND = {
    "k_gas_W_mK": {"BAHADORI_G": (0.933, 49.9, 68), "CHUNG": (0.0835, 0.259, 82), "COOLPROP": (0.0166, 0.0924, 45),
        "DIPPR_9B": (0.111, 0.231, 82), "DIPPR_PERRY_8E": (0.038, 0.115, 64), "ELI_HANLEY": (0.159, 0.344, 82),
        "EUCKEN": (0.233, 0.283, 82), "EUCKEN_MOD": (0.145, 0.402, 82), "GHARAGHEIZI_G": (0.525, 7.49, 82),
        "REFPROP_FIT": (0.0139, 0.0779, 82), "VDI_PPDS": (0.0626, 0.482, 78), "VDI_TABULAR": (0.399, 0.894, 27)},
    "k_liquid_W_mK": {"BAHADORI_L": (1.05, 2.57, 68), "COOLPROP": (0.00026, 0.000779, 75),
        "DIPPR_PERRY_8E": (0.0519, 0.182, 53), "GHARAGHEIZI_L": (0.404, 0.807, 52), "LAKSHMI_PRASAD": (2.76, 9.86, 51),
        "NICOLA": (0.527, 2.56, 75), "NICOLA_ORIGINAL": (0.704, 2.15, 75), "REFPROP_FIT": (0.00134, 0.00609, 57),
        "SATO_RIEDEL": (0.627, 8.5, 75), "SHEFFY_JOHNSON": (2.74, 8.67, 75), "VDI_PPDS": (0.0764, 0.274, 65),
        "VDI_TABULAR": (0.145, 0.319, 42)},
    "mu_gas_Pa_s": {"COOLPROP": (0.00599, 0.0263, 59), "DIPPR_PERRY_8E": (0.0315, 0.0715, 52),
        "GHARAGHEIZI": (0.217, 0.4, 58), "LUCAS_GAS": (0.0353, 0.0736, 54), "REFPROP_FIT": (0.00593, 0.0126, 59),
        "STIEL_THODOS": (0.21, 0.372, 59), "VDI_PPDS": (0.0779, 1.35, 49), "VDI_TABULAR": (0.238, 0.555, 19),
        "YOON_THODOS": (0.221, 0.355, 59)},
    "mu_liquid_Pa_s": {"COOLPROP": (0.000962, 0.00631, 55), "DIPPR_PERRY_8E": (0.0794, 0.483, 42),
        "DUTT_PRASAD": (0.0152, 0.0413, 6), "JOBACK": (0.781, 115, 22), "LETSOU_STIEL": (0.575, 0.995, 55),
        "PRZEDZIECKI_SRIDHAR": (63.1, 653, 42), "REFPROP_FIT": (0.00143, 0.00631, 55), "VDI_PPDS": (0.129, 1.02, 45),
        "VDI_TABULAR": (0.119, 0.352, 27), "VISWANATH_NATARAJAN_2": (5.29e+04, 6.76e+04, 4),
        "VISWANATH_NATARAJAN_3": (0.0369, 0.0414, 4)},
    "sigma_N_m": {"Aleem": (0.944, 0.985, 30), "BROCK_BIRD": (0.18, 1.22, 81), "JASPER": (0.0225, 0.0749, 12),
        "MIQUEU": (0.153, 1.42, 81), "PITZER_SIGMA": (0.219, 1.31, 81), "REFPROP": (0, 0, 74),
        "REFPROP_FIT": (9.19e-05, 0.00415, 81), "SASTRI_RAO": (0.197, 1.17, 81), "SOMAYAJULU": (0.0737, 0.208, 63),
        "SOMAYAJULU2": (0.0497, 0.208, 63), "VDI_PPDS": (0.0914, 0.237, 73), "VDI_TABULAR": (0.0678, 0.215, 51),
        "ZUO_STENBY": (0.139, 1.17, 81)},
}
_TRANSPORT_KEYS = {"mu_liquid_Pa_s": ("mu", "liquid"), "mu_gas_Pa_s": ("mu", "gas"), "k_liquid_W_mK": ("k", "liquid"),
                   "k_gas_W_mK": ("k", "gas"), "sigma_N_m": ("sigma", "liquid")}


@lru_cache(maxsize=128)
def _coolprop_fluid(cas: str) -> dict | None:
    """CoolProp fluid for a CAS number and the kind of transport correlation it carries: 'reference' (a dedicated
    correlation published for that fluid and fitted to its data) or 'ECS' / 'Chung' / 'rhosr-CSS' (corresponding states,
    never used here) or 'none'."""
    try:
        import json
        import CoolProp
        import CoolProp.CoolProp as CP
        name = CP.get_fluid_param_string(cas, "name")
        j = json.loads(CP.get_fluid_param_string(name, "JSON"))
        j = j[0] if isinstance(j, list) else j
        tc, tmax, pmax = CP.PropsSI("Tcrit", name), CP.PropsSI("Tmax", name), CP.PropsSI("pmax", name)
        tmin = max(CP.PropsSI("Tmin", name), CP.PropsSI("T_triple", name))
    except Exception:  # noqa: BLE001 — not a CoolProp fluid
        return None
    transport = j.get("TRANSPORT") or {}
    kinds = {}
    for short, key in (("mu", "viscosity"), ("k", "conductivity")):
        x = transport.get(key)
        if isinstance(x, dict) and x and not x.get("type"):
            kinds[short] = ("reference", x.get("BibTeX") or x.get("hardcoded"))
        elif isinstance(x, (dict, list)) and x:
            first = x[0] if isinstance(x, list) else x
            kinds[short] = (str(first.get("type")), first.get("BibTeX"))
        else:
            kinds[short] = ("none", None)
    st = (j.get("ANCILLARIES") or {}).get("surface_tension") or {}
    kinds["sigma"] = ("reference", st.get("BibTeX") or "Mulero") if st else ("none", None)
    return {"name": name, "cas": CP.get_fluid_param_string(name, "CAS"), "kinds": kinds, "Tc": tc, "Tmin": tmin,
            "Tmax": tmax, "pmax": pmax,
            "version": CoolProp.__version__}


def _coolprop_value(fl: dict, what: str, state: str, T: float, P: float | None = None) -> float | None:
    """CoolProp transport value: state 'liquid' = saturated liquid, 'dilute' = zero-density gas, 'gas' = gas at T, P."""
    import CoolProp.CoolProp as CP
    try:
        st = CP.AbstractState("HEOS", fl["name"])
        if state == "liquid":
            st.update(CP.QT_INPUTS, 0.0, T)
        elif state == "dilute":
            st.update(CP.DmolarT_INPUTS, DILUTE_RHO_MOL_M3, T)
        else:
            st.update(CP.PT_INPUTS, P, T)
            if st.phase() in (CP.iphase_liquid, CP.iphase_twophase):
                return None
        v = st.viscosity() if what == "mu" else st.conductivity() if what == "k" else st.surface_tension()
    except Exception:  # noqa: BLE001 — outside the correlation or the EOS
        return None
    return float(v) if math.isfinite(v) and v > 0 else None


def _nist_band(fl: dict, what: str, T: float) -> float | None:
    if what == "sigma":
        b = _CP_SIGMA_VS_NIST.get(fl["cas"])
        return None if b is None else b[0 if T <= NEAR_CRITICAL_TR * fl["Tc"] else 1]
    b = _CP_VS_NIST.get(fl["cas"])
    return None if b is None else b[0 if what == "mu" else 1]


def _add_flag(d: dict, text: str) -> None:
    d["flag"] = f"{d['flag']}; {text}" if d.get("flag") else text


def _transport(key: str, p: dict, cas: str, T: float, tc: float | None, sigma_obj=None) -> dict:
    """Saturated-liquid / dilute-gas transport value: CoolProp's reference correlation when it matched NIST WebBook
    (measured), else thermo's value cross-checked against that correlation; plus the measured band of thermo's method."""
    what, phase = _TRANSPORT_KEYS[key]
    out = dict(p)
    band = _METHOD_BAND.get(key, {}).get(p.get("method"))
    if p.get("value") is not None and band:
        out["measured_band_rel"] = {"p90": band[0], "max": band[1], "n_points": band[2]}
        if band[0] > METHOD_BAND_FLAG:
            _add_flag(out, f"en la validación contra correlaciones de referencia este método ({p['method']}) se apartó más "
                           f"de un {100 * band[0]:.0f} % en el 10 % peor de {band[2]} puntos (máx. {100 * band[1]:.0f} %)")
    fl = _coolprop_fluid(cas)
    kind, bib = fl["kinds"][what] if fl else ("none", None)
    if kind != "reference":
        out["reference_correlation"] = None
        if p.get("value") is not None:
            out["note"] = (out.get("note") or "") + ("; CoolProp no tiene correlación de referencia para este compuesto"
                                                    if kind == "none" else
                                                    f"; CoolProp solo tiene un modelo de estados correspondientes ({kind})")
    else:
        inside = fl["Tmin"] <= T < fl["Tc"] if phase == "liquid" else fl["Tmin"] <= T <= fl["Tmax"]
        ref = _coolprop_value(fl, what, "liquid" if phase == "liquid" else "dilute", T) if inside else None
        nist = _nist_band(fl, what, T)
        # sigma: Mulero's correlation matched NIST in 18 of 19 fluids (water uses IAPWS), so it is used for every fluid
        use = ref is not None and p.get("method") not in _THERMO_PREFERRED.get(key, ()) and (
            what == "sigma" or (nist is not None and nist <= NIST_AGREE_TOL))
        state_txt = "líquido saturado" if phase == "liquid" else "límite de gas diluido"
        nist_txt = (f"; contra NIST WebBook se apartó como máximo un {100 * nist:.2g} % (medido)" if nist is not None
                    else "; no contrastada con NIST WebBook en esta herramienta")
        if use:
            lim = [float(fl["Tmin"]), float(fl["Tc"] if phase == "liquid" else fl["Tmax"])]
            data_lim = None
            if what == "sigma" and sigma_obj is not None and "REFPROP" in dict(sigma_obj.T_limits):
                data_lim = [float(x) for x in dict(sigma_obj.T_limits)["REFPROP"]]
            new = {"value": ref, "method": "COOLPROP_REFERENCE", "basis": "data", "estimated": False, "T_range_K": lim,
                   "in_range": True, "extrapolated": False, "reference_correlation": bib, "nist_webbook_max_dev_rel": nist,
                   "note": f"correlación de referencia {bib} ({state_txt}) evaluada con CoolProp {fl['version']}" + nist_txt,
                   "thermo_value": p.get("value"), "thermo_method": p.get("method"),
                   "thermo_vs_reference_rel": p["value"] / ref - 1.0 if p.get("value") else None}
            for k in ("n_data_methods", "data_methods_spread_rel", "data_methods_median", "data_methods_farthest"):
                if k in p:
                    new[k] = p[k]
            if "data_methods_median" in p:
                new["value_vs_data_median_rel"] = ref / p["data_methods_median"] - 1.0
            if "alternatives" in p:
                new["alternatives"] = list(p["alternatives"])
                if p.get("value") is not None:
                    new["alternatives"].insert(0, {"method": p["method"], "value": p["value"], "basis": p.get("basis")})
            if data_lim and not data_lim[0] <= T <= data_lim[1]:
                new["data_range_K"] = data_lim
                _add_flag(new, f"T fuera del intervalo de datos de la correlación de Mulero ({data_lim[0]:.1f}-"
                               f"{data_lim[1]:.1f} K): se extrapola hacia σ(Tc) = 0, como hace REFPROP")
            out = new
        elif ref is not None and p.get("value") is not None:
            dev = p["value"] / ref - 1.0
            out["reference_check"] = {"method": "COOLPROP_REFERENCE", "correlation": bib, "value": ref, "deviation_rel": dev,
                                      "nist_webbook_max_dev_rel": nist,
                                      "note": f"correlación de referencia ({state_txt}) evaluada con CoolProp" + nist_txt}
            if abs(dev) > TRANSPORT_CHECK_TOL:
                why = (f"esa correlación de CoolProp se apartó de NIST WebBook hasta un {100 * nist:.0f} % para este "
                       "fluido, así que la diferencia no indica que el valor dado sea el malo"
                       if nist is not None and nist > NIST_AGREE_TOL else
                       "ninguna de las dos se contrastó con NIST WebBook aquí: tomar la diferencia como incertidumbre")
                _add_flag(out, f"difiere un {100 * dev:+.1f} % de la correlación de referencia {bib} (CoolProp); {why}")
    if (phase == "liquid" and tc and T > NEAR_CRITICAL_TR * tc and out.get("value") is not None):
        _add_flag(out, f"cerca del punto crítico (T/Tc = {T / tc:.3f}): las fuentes independientes discrepan hasta un 15 % "
                       "(medido en la validación)")
    return out


def _gas_transport_at_P(what: str, cas: str, T: float, P: float, dilute: float | None) -> dict:
    """Gas viscosity / conductivity at T, P (dense gas) with CoolProp's reference correlation (dilute + residual +
    critical terms); none is computed without one."""
    fl = _coolprop_fluid(cas)
    kind, bib = fl["kinds"][what] if fl else ("none", None)
    if kind != "reference":
        return {"value": None, "method": None, "in_range": False,
                "note": "sin correlación de referencia en CoolProp para este compuesto: no hay corrección de presión "
                        "validada; el valor disponible es el del gas diluido"}
    if not (fl["Tmin"] <= T <= fl["Tmax"] and P <= fl["pmax"]):
        return {"value": None, "method": "COOLPROP_REFERENCE", "in_range": False,
                "note": f"T, P fuera del intervalo de la ecuación de estado de CoolProp ({fl['Tmin']:.1f}-"
                        f"{fl['Tmax']:.1f} K, "
                        f"hasta {fl['pmax']:.4g} Pa)"}
    v = _coolprop_value(fl, what, "gas", T, P)
    if v is None:
        return {"value": None, "method": "COOLPROP_REFERENCE", "in_range": False,
                "note": "CoolProp ubica este estado como líquido o bifásico (su presión de vapor difiere de la de thermo) o "
                        "la correlación no converge"}
    nist = _nist_band(fl, what, T)
    out = {"value": v, "method": "COOLPROP_REFERENCE", "basis": "data", "estimated": False,
           "T_range_K": [float(fl["Tmin"]), float(fl["Tmax"])], "in_range": True, "extrapolated": False, "P_Pa": P,
           "reference_correlation": bib, "nist_webbook_max_dev_rel": nist, "dilute_value": dilute,
           "pressure_effect_rel": v / dilute - 1.0 if dilute else None,
           "note": f"correlación de referencia {bib} a T y P (términos diluido, residual y crítico) vía CoolProp "
                   f"{fl['version']}"}
    if nist is None:
        out["flag"] = "correlación de referencia de CoolProp no contrastada con NIST WebBook en esta herramienta"
    elif nist > NIST_AGREE_TOL:
        out["flag"] = (f"contra NIST WebBook esta correlación de CoolProp se apartó hasta un {100 * nist:.0f} % para este "
                       "fluido")
    return out


@contextmanager
def _thermo_quiet():
    """thermosteam (BioSTEAM, used by the costing engines) flips thermo's global RAISE_PROPERTY_CALCULATION_ERROR to True,
    which makes thermo raise for a property with no method (e.g. the sublimation enthalpy of R134a). This module expects
    thermo's default (None for a missing method, reported as such), so the flag is pinned to False here and restored."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    saved = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
    try:
        yield
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = saved


def _thermo_defaults(func):
    @wraps(func)
    def inner(*args, **kwargs):
        with _thermo_quiet():
            return func(*args, **kwargs)
    return inner


@lru_cache(maxsize=64)
def _package(cas: str):
    from thermo import ChemicalConstantsPackage
    return ChemicalConstantsPackage.from_IDs([cas])


@_thermo_defaults
def compound_properties(compound: str, T_K: float | None = None, P_Pa: float | None = None,
                        alternatives: bool = False) -> dict:
    """Identity, constants and (if T_K is given) temperature-dependent properties of one pure compound, each with the
    library method that produced it, its basis (data / compilation / derived / estimate) and its validity range.

    P_Pa (needs T_K) only decides the stable phase against the vapour pressure and gives the gas density at T, P; liquid
    properties are those of the saturated / low-pressure liquid. `alternatives=True` lists every other method valid at T."""
    import chemicals
    import thermo
    meta, alias_note = identify(compound)
    if T_K is not None and (not isinstance(T_K, (int, float)) or not math.isfinite(T_K) or T_K <= 0):
        raise ValueError("T_K debe ser una temperatura absoluta positiva en kelvin.")
    if P_Pa is not None and (not isinstance(P_Pa, (int, float)) or not math.isfinite(P_Pa) or P_Pa <= 0):
        raise ValueError("P_Pa debe ser una presión absoluta positiva en pascales.")
    if P_Pa is not None and T_K is None:
        raise ValueError("P_Pa se dio sin T_K: la presión solo sirve para decidir la fase y la densidad del gas a una "
                         "temperatura; da también T_K.")
    cas = meta.CASs
    pkg, corr = _package(cas)
    warnings: list[str] = []
    assumptions = [
        "Todos los valores salen de las bases de datos y correlaciones de chemicals y thermo; ninguno se midió ni se ajustó "
        "aquí.",
        "El método indicado en cada valor es el primero del orden de preferencia de la librería que cubre la temperatura "
        "pedida (regla en el campo method).",
        "basis = 'data' significa valor medido, compilación evaluada o correlación ajustada a datos o a una ecuación de "
        "estado de referencia; no es una incertidumbre: la dispersión entre fuentes está en data_sources_spread / "
        "data_methods_spread_rel.",
        "Viscosidad, conductividad y tensión superficial: si CoolProp tiene una correlación de referencia propia del "
        "compuesto (no de estados correspondientes) y esa correlación coincidió con NIST WebBook dentro del 1 % en la "
        "validación (tests/test_transport_properties.py), se usa ella (method COOLPROP_REFERENCE, reference_correlation); "
        "si no, se da el valor de thermo y, cuando existe, la correlación de CoolProp como reference_check. La tensión "
        "superficial usa la correlación de Mulero y otros vía CoolProp en todo compuesto que la tenga (coincidió con NIST "
        "en 18 de 19 fluidos; en los no contrastados, como etanol o acetona, se usa igual), salvo en el agua (IAPWS). El "
        "intervalo de T de CoolProp es el de su ecuación de estado; CoolProp no publica el intervalo de validez de cada "
        "correlación de transporte.",
    ]
    by = _identified_by(compound, meta)
    identity = {"query": compound, "name": meta.common_name, "iupac_name": meta.iupac_name, "CAS": cas,
                "formula": meta.formula, "molar_mass_kg_mol": float(meta.MW) / 1000.0, "smiles": meta.smiles,
                "inchi_key": meta.InChI_key, "identified_by": by}
    if alias_note:
        identity["alias"] = alias_note
    if by == "formula":
        warnings.append(f"«{compound}» se identificó por fórmula y puede tener isómeros; la base de datos devolvió "
                        f"{meta.common_name} (CAS {cas}). Usa el nombre o el CAS para evitar ambigüedad.")
    constants = _constants(cas, pkg)
    tc, pc = constants["Tc_K"]["value"], constants["Pc_Pa"]["value"]
    tb, tm, omega = constants["Tb_K"]["value"], constants["Tm_K"]["value"], constants["acentric_factor"]["value"]
    vp, hv = corr.VaporPressures[0], corr.EnthalpyVaporizations[0]
    if tb is not None and (tc is None or tb < tc):
        h_tb = _tdep(hv, tb, False)
        if h_tb.get("value") is not None:
            h_tb["note"] += f"; evaluado en Tb = {tb:.2f} K ({constants['Tb_K']['method']})"
        constants["Hvap_Tb_J_mol"] = h_tb
    for key, c in constants.items():
        if c.get("value") is not None and c.get("flag") and c.get("basis") in ("estimate", "unclassified", "compilation"):
            warnings.append(f"{key}: {c['flag']} (método {c['method']}).")

    evidence: dict = {"verified": False, "checks": {}, "reasons": []}
    checks = evidence["checks"]
    lim = vp.T_limits.get(vp.method) if vp.method else None
    if tb is not None and lim and lim[0] <= tb <= lim[1]:
        p_tb = _calc(vp, tb, vp.method)
        if p_tb:
            dev = p_tb / P_ATM - 1.0
            checks["psat_at_Tb_rel_dev"] = {"value": dev, "tolerance": PSAT_TB_TOL, "ok": abs(dev) <= PSAT_TB_TOL,
                                            "what": f"Psat({vp.method}) en Tb ({constants['Tb_K']['method']}) frente a "
                                                    "101325 Pa"}
    if None not in (tc, pc, omega) and lim and lim[0] <= 0.7 * tc <= lim[1]:
        p07 = _calc(vp, 0.7 * tc, vp.method)
        if p07:
            dev = -math.log10(p07 / pc) - 1.0 - omega
            checks["acentric_definition_abs_dev"] = {
                "value": dev, "tolerance": OMEGA_TOL, "ok": abs(dev) <= OMEGA_TOL,
                "what": f"ω de la definición con Psat({vp.method}) a 0.7·Tc menos ω tabulado "
                        f"({constants['acentric_factor']['method']})"}

    if None not in (tc, pc) and lim and lim[0] <= tc * (1 - 1e-5) <= lim[1] * (1 + 1e-9):
        p_tc = _calc(vp, min(tc * (1 - 1e-5), lim[1]), vp.method)
        if p_tc:
            dev = p_tc / pc - 1.0
            checks["psat_at_Tc_rel_dev"] = {"value": dev, "tolerance": PSAT_TC_TOL, "ok": abs(dev) <= PSAT_TC_TOL,
                                            "what": f"Psat({vp.method}) en Tc ({constants['Tc_K']['method']}) frente a Pc "
                                                    f"({constants['Pc_Pa']['method']})"}

    out = {"identity": identity, "constants": constants}
    not_included = [
        "Efecto de la presión sobre las propiedades del líquido (densidad, viscosidad, Cp, conductividad): se dan las del "
        "líquido saturado o a baja presión.",
        "Propiedades del sólido y presión de sublimación.",
        "Viscosidad y conductividad del gas a alta presión para compuestos sin correlación de referencia en CoolProp (p. "
        "ej. acetona): solo se da la del gas diluido (mu_gas_Pa_s, k_gas_W_mK); con P_Pa y correlación de referencia se "
        "dan además mu_gas_at_P_Pa_s y k_gas_at_P_W_mK.",
        "Incertidumbre propia de cada correlación de referencia (la que declaran sus autores): solo se informa la "
        "desviación medida frente a NIST WebBook.",
        "Incertidumbre de cada dato: la librería no la trae; solo se informa la dispersión entre fuentes.",
        "Mezclas: esta herramienta es de compuesto puro (para mezclas, independent_flash).",
    ]
    if T_K is not None:
        T = float(T_K)
        above_tc = tc is not None and T >= tc
        no_liq = f"T = {T:g} K ≥ Tc = {tc:.2f} K: no existe equilibrio líquido-vapor ni líquido" if above_tc else None
        props = {
            "Psat_Pa": _tdep(vp, T, alternatives, no_liq, "Psat_Pa"),
            "Hvap_J_mol": _tdep(hv, T, alternatives, no_liq, "Hvap_J_mol"),
            "Cp_liquid_J_molK": _tdep(corr.HeatCapacityLiquids[0], T, alternatives, no_liq, "Cp_liquid_J_molK"),
            "Cp_ideal_gas_J_molK": _tdep(corr.HeatCapacityGases[0], T, alternatives, None, "Cp_ideal_gas_J_molK"),
            "V_liquid_m3_mol": _tdep(corr.VolumeLiquids[0], T, alternatives, no_liq, "V_liquid_m3_mol"),
            "mu_liquid_Pa_s": _tdep(corr.ViscosityLiquids[0], T, alternatives, no_liq, "mu_liquid_Pa_s"),
            "mu_gas_Pa_s": _tdep(corr.ViscosityGases[0], T, alternatives, None, "mu_gas_Pa_s"),
            "k_liquid_W_mK": _tdep(corr.ThermalConductivityLiquids[0], T, alternatives, no_liq, "k_liquid_W_mK"),
            "k_gas_W_mK": _tdep(corr.ThermalConductivityGases[0], T, alternatives, None, "k_gas_W_mK"),
            "sigma_N_m": _tdep(corr.SurfaceTensions[0], T, alternatives, no_liq, "sigma_N_m"),
        }
        for key, (_, phase) in _TRANSPORT_KEYS.items():
            if not (above_tc and phase == "liquid"):
                props[key] = _transport(key, props[key], cas, T, tc, corr.SurfaceTensions[0])
        mw = identity["molar_mass_kg_mol"]
        vl = props["V_liquid_m3_mol"]
        rho_l = dict(vl)
        if vl.get("value") is not None:
            rho_l["value"] = mw / vl["value"]
            rho_l.pop("alternatives", None)
        props["rho_liquid_kg_m3"] = rho_l
        psat = props["Psat_Pa"].get("value")
        state = {"T_K": T, "P_Pa": float(P_Pa) if P_Pa is not None else None}
        if tm is not None and T < tm:
            warnings.append(f"T = {T:g} K está por debajo del punto de fusión normal ({tm:.2f} K, "
                            f"{constants['Tm_K']['method']}): el compuesto es sólido; las propiedades de líquido "
                            "corresponden a líquido subenfriado o a correlaciones fuera de intervalo.")
        if P_Pa is not None:
            P = float(P_Pa)
            if above_tc:
                state["phase"] = "supercritical" if pc is not None and P >= pc else "gas"
                state["basis"] = "T ≥ Tc" + (" y P ≥ Pc" if state["phase"] == "supercritical" else " y P < Pc")
            elif tm is not None and T < tm:
                state["phase"] = "solid"
                state["basis"] = "T < Tm normal (no se considera el efecto de la presión sobre la fusión)"
            elif psat is not None:
                state["phase"] = "liquid" if P > psat * (1 + 1e-9) else ("gas" if P < psat * (1 - 1e-9) else "saturated")
                state["basis"] = f"P frente a Psat(T) = {psat:.6g} Pa ({props['Psat_Pa']['method']})"
                if props["Psat_Pa"].get("extrapolated") or props["Psat_Pa"].get("estimated"):
                    warnings.append("La fase se decidió con una presión de vapor estimada o extrapolada.")
            else:
                state["phase"] = None
                state["basis"] = "sin presión de vapor no se puede decidir la fase"
            if state["phase"] in ("gas", "supercritical", "saturated"):
                vg = _gas_volume(corr.VolumeGases[0], T, P)
                props["V_gas_m3_mol"] = vg
                rho_g = dict(vg)
                if vg.get("value") is not None:
                    rho_g["value"] = mw / vg["value"]
                    if state["phase"] == "supercritical" and vg["method"] != "COOLPROP":
                        rho_g["flag"] = vg["flag"] = ("VALOR NO FIABLE: estado supercrítico sin ecuación de estado de "
                                                      "referencia")
                props["rho_gas_kg_m3"] = rho_g
                if state["phase"] != "saturated":
                    for key, what, dil in (("mu_gas_at_P_Pa_s", "mu", "mu_gas_Pa_s"),
                                           ("k_gas_at_P_W_mK", "k", "k_gas_W_mK")):
                        g = _gas_transport_at_P(what, cas, T, P, props[dil].get("value"))
                        props[key] = g
                        eff = g.get("pressure_effect_rel")
                        vc = constants["Vc_m3_mol"]["value"]
                        rho_r = vc / vg["value"] if vc and vg.get("value") else None
                        if eff is not None and abs(eff) > TRANSPORT_CHECK_TOL:
                            warnings.append(f"{dil} es la del gas diluido; a P = {P:.4g} Pa la correlación de referencia da "
                                            f"{key} = {g['value']:.4g} ({100 * eff:+.1f} %): usa {key}.")
                        elif g.get("value") is None and (rho_r is None or rho_r > DENSE_GAS_RHO_R):
                            dens = f"ρ/ρc = {rho_r:.3f}" if rho_r is not None else "sin Vc no se puede estimar la densidad"
                            warnings.append(f"{key}: no disponible ({g['note']}). {dens}: {dil} (gas diluido) puede no "
                                            "valer a esta presión; en la validación (19 fluidos a 5-10 MPa) el valor "
                                            "diluido quedó por debajo del de NIST hasta un 80 %, más cuanto más denso es "
                                            "el gas.")
            else:
                props["rho_gas_kg_m3"] = {"value": None, "method": None,
                                          "note": f"a T y P el compuesto es {state['phase']}: "
                                                                                 "no hay gas"}
        out["state"] = state
        out["properties_at_T"] = props
        for key, p in props.items():
            if key == "V_liquid_m3_mol" or p.get("value") is None:
                continue
            if p.get("flag"):
                warnings.append(f"{key}: {p['flag']} (método {p['method']}).")
            if p.get("reference_correlation"):
                continue        # reference correlation checked against NIST: the spread of thermo's tables is informative
            if abs(p.get("value_vs_data_median_rel", 0.0)) > SPREAD_WARN and p.get("n_data_methods", 0) >= 3:
                warnings.append(f"{key}: el valor elegido ({p['method']}) se aparta un "
                                f"{100 * p['value_vs_data_median_rel']:+.1f} % de la mediana de los {p['n_data_methods']} "
                                "métodos basados en datos válidos a esta T: revísalo con alternatives=True antes de usarlo.")
            elif p.get("data_methods_spread_rel", 0.0) > SPREAD_WARN:
                far = p["data_methods_farthest"]
                warnings.append(f"{key}: los {p['n_data_methods']} métodos basados en datos válidos a esta T abarcan un "
                                f"{100 * p['data_methods_spread_rel']:.1f} % (el más alejado es {far['method']} = "
                                f"{far['value']:.4g}); "
                                + ("el valor elegido coincide con la mediana." if p["n_data_methods"] >= 3 else
                                   "con solo dos fuentes no se puede decir cuál es la buena."))
        # Clapeyron: ΔHvap = T·(Vg − Vl)·dPsat/dT with the saturated volumes
        hvap, vliq = props["Hvap_J_mol"], props["V_liquid_m3_mol"]
        if (not above_tc and psat and hvap.get("value") and vliq.get("value") and props["Psat_Pa"].get("in_range")
                and (tc is None or T <= CLAPEYRON_MAX_TR * tc)):
            vg_sat = _gas_volume(corr.VolumeGases[0], T, psat * (1.0 - SAT_MARGIN))
            try:
                dpdt = float(vp.calculate_derivative(T, props["Psat_Pa"]["method"]))
            except Exception:  # noqa: BLE001
                dpdt = None
            if vg_sat.get("value") and dpdt and dpdt > 0:
                h_clap = T * (vg_sat["value"] - vliq["value"]) * dpdt
                dev = hvap["value"] / h_clap - 1.0
                counted = vg_sat["basis"] == "data" and hvap["basis"] == "data" and props["Psat_Pa"]["basis"] == "data" \
                    and vliq["basis"] == "data" and hvap["in_range"] and vliq["in_range"]
                checks["clapeyron_Hvap_rel_dev"] = {
                    "value": dev, "tolerance": CLAPEYRON_TOL, "ok": abs(dev) <= CLAPEYRON_TOL,
                    "counted_for_verified": counted,
                    "Hvap_clapeyron_J_mol": h_clap,
                    "what": f"ΔHvap ({hvap['method']}) frente a T·(Vg − Vl)·dPsat/dT con Psat {props['Psat_Pa']['method']}, "
                            f"Vg {vg_sat['method']} y Vl {vliq['method']}"}
                if not counted:
                    checks["clapeyron_Hvap_rel_dev"]["note"] = ("informativo: alguno de los cuatro ingredientes es una "
                                                                "estimación o está extrapolado")
                    if abs(dev) > 0.10:
                        warnings.append(
                            f"ΔHvap ({hvap['method']}) difiere un {100 * dev:+.0f} % de la ecuación de Clapeyron con el "
                            f"volumen de vapor de {vg_sat['method']}: alguno de los ingredientes es una estimación o una "
                            "extrapolación, o el vapor se asocia (ácidos carboxílicos, HF) y el virial no lo describe; en "
                            "ese último caso el ΔHvap dado es el de vaporización al vapor real, no el ΔvapH° estándar al "
                            "gas ideal monómero.")
    if T_K is not None:
        diffs = []
        for key in _TRANSPORT_KEYS:
            p = out["properties_at_T"].get(key) or {}
            d = p.get("thermo_vs_reference_rel")
            d = (p.get("reference_check") or {}).get("deviation_rel") if d is None else d
            if d is not None:
                diffs.append((abs(d), key))
        if diffs:
            worst = max(diffs)
            checks["transport_thermo_vs_reference_max_rel"] = {
                "value": worst[0], "tolerance": TRANSPORT_CHECK_TOL, "ok": worst[0] <= TRANSPORT_CHECK_TOL,
                "counted_for_verified": False, "property": worst[1],
                "what": "mayor |thermo / correlación de referencia de CoolProp − 1| entre las propiedades de transporte",
                "note": "informativo: dos fuentes de referencia que discrepan no indican cuál es la buena (ver flags)"}
    counted = {k: c for k, c in checks.items() if c.get("counted_for_verified", True)}
    if not counted:
        evidence["reasons"].append("No hay comprobaciones cruzadas posibles: faltan Tb, constantes críticas o una presión "
                                   "de vapor que cubra Tb, Tc o 0.7·Tc.")
    for k, c in counted.items():
        if not c["ok"]:
            evidence["reasons"].append(f"{k} = {c['value']:.4g} supera la tolerancia {c['tolerance']:g}: {c['what']}.")
    evidence["verified"] = bool(counted) and not evidence["reasons"]
    evidence["meaning"] = ("verified certifica que los datos de la librería para este compuesto son coherentes entre sí "
                           "(Psat-Tb, Psat-Pc, ω-Psat, Clapeyron); no certifica cada valor contra una medición.")
    if not evidence["verified"]:
        warnings.append("Los datos de la librería para este compuesto no superan las comprobaciones cruzadas (ver "
                        "evidence).")
    out.update(
        warnings=warnings, assumptions=assumptions, not_included=not_included,
        method=(f"Consulta de chemicals {chemicals.__version__} (constantes) y thermo {thermo.__version__} (propiedades "
                "dependientes de T). Método por propiedad: el predeterminado de thermo si T está en su intervalo; si no, el "
                "mejor clasificado basado en datos que cubra T; si no, el mejor clasificado de cualquier tipo (estimación, "
                "marcada); si no, el predeterminado extrapolado (marcado). Transporte: correlación de referencia de "
                "CoolProp (AbstractState HEOS: líquido saturado, gas a 1e-6 mol/m3 o gas a T y P) donde coincidió con "
                "NIST."),
        evidence=evidence)
    return out
