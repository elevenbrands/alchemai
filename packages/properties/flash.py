"""Phase-equilibrium flash on `thermo` (MIT, ChEDL), independent of the DWSIM engine: a second opinion (bloque `props`).

What it answers: which phases exist at a specification (T-P, P-H, P-S, P-VF or T-VF), their fractions, compositions and
K-values, and whether a second liquid appears (VLLE). The equations are solved by `thermo.flash.FlashVLN` (two liquid
candidates + gas, with thermo's own Michelsen stability tests); P-H and P-S are an outer bracketed root search on T over
T-P flashes written here, so that they are VLLE-aware for every model. P-VF / T-VF use thermo's two-phase solver and, when
its state fails the independent checks (typically a hidden liquid split), a bisection in T (or P) over the same T-P
flashes. When the specification falls on the isothermal jump of a binary three-phase line (heteroazeotrope), the phases
of both sides of the jump are evaluated at the jump and their fractions solved from the material balance plus the
specification (non-negative least squares): solver "own_three_phase_jump_resolution".

Thermodynamic model: NO default. The caller names it:
  * "PR" / "SRK"                  cubic equation of state for all phases (φ-φ), van der Waals one-fluid mixing, kij;
  * "NRTL" / "UNIQUAC" / "WILSON" activity coefficients with the caller's binary parameters (tau_ij = a_ij + b_ij/T,
                                  the same convention as packages.vle.models), ideal gas, γ·x·Psat = y·P (no Poynting);
  * "UNIFAC" / "UNIFAC_DORTMUND" / "UNIFAC_LLE"  predictive group contribution (tables shipped with thermo);
  * "IDEAL"                       Raoult's law (γ = 1, ideal gas), only as an explicit choice.
Every binary parameter is reported with its origin: "user", "library" (ChemSep PR table of thermo's IPDB, only for PR) or
"zero_by_absence" (warned).

Evidence computed here, independently of thermo's solver, on the returned state:
  * material balance over the phases and normalisation of each composition;
  * equality of fugacities between phases, max |ln f_i(phase) − ln f_i(reference phase)|;
  * Rachford-Rice residual of each non-reference phase with the K-values of the solution and, for two phases, the
    phase fraction re-solved from those K-values by `rachford_rice` (own bracketed solver);
  * a Michelsen (1982) tangent-plane-distance test by successive substitution written here, started from Wilson
    vapour-like and liquid-like estimates and from every pure component: a trial with TPD < 0 that is not one of the
    phases found means the state is NOT stable (verified = False);
  * the residual of the specification (H, S or vapour fraction) for non T-P flashes.
`verified` certifies the numerics of the run for the model chosen, not that the model represents the mixture."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from packages.properties.pure import _basis_const, _basis_tdep, identify

R_GAS = 8.314462618
MODELS_EOS = ("PR", "SRK")
MODELS_GE = ("NRTL", "UNIQUAC", "WILSON")
MODELS_UNIFAC = ("UNIFAC", "UNIFAC_DORTMUND", "UNIFAC_LLE")
MODELS = MODELS_EOS + MODELS_GE + MODELS_UNIFAC + ("IDEAL",)
BALANCE_TOL = 1e-8          # material balance and normalisation, mole fraction
FUGACITY_TOL = 1e-4         # max |Δ ln f| between phases, 0.01 % in fugacity (thermo's bubble/dew solvers stop near 2e-5;
                            # its T-P flash reaches 1e-7 or better)
RR_BETA_TOL = 1e-4          # phase fraction re-solved by Rachford-Rice from the K-values vs the flash
RR_TOL = 1e-8               # Rachford-Rice residual
TPD_TOL = 1e-8              # a trial phase with tm < −max(TPD_TOL, 10·fugacity residual) makes the state unstable
TPD_MAXITER = 500           # successive substitution iterations per trial
TPD_XTOL = 1e-11            # convergence of ln W in the stability test
SS_MAXITER = 5000           # multiphase successive substitution (fallback solver)
SS_XTOL = 1e-11             # convergence of ln x in the fallback solver
TRIVIAL_TOL = 1e-5          # a trial closer than this (max |Δx|) to an existing phase is that phase
SPEC_TOL = 1e-6             # relative residual of the H / S / VF specification
T_BRACKET = (1.12, 40)      # geometric step and maximum expansions of the T bracket for P-H and P-S
WILSON_VOLUME_T = 298.15    # K, temperature of the liquid molar volumes of the Wilson model (as packages.vle.tool)
ENTHALPY_REFERENCE = ("H = 0 y S = 0 para cada componente puro como gas ideal a 298.15 K y 101325 Pa (referencia de thermo; "
                      "no son entalpías de formación: sirven para diferencias entre estados de la misma mezcla)")


# ---- inputs -----------------------------------------------------------------------------------------------------------
def _require_model(model) -> str:
    if model is None or not isinstance(model, str) or not model.strip():
        raise ValueError("Falta 'model': el modelo termodinámico no tiene valor por defecto porque decide el resultado. "
                         "Opciones: 'PR' o 'SRK' (ecuación cúbica para todas las fases, con kij), 'NRTL', 'UNIQUAC' o "
                         "'WILSON' (coeficientes de actividad con los parámetros binarios que tú des en "
                         "activity_parameters), 'UNIFAC', 'UNIFAC_DORTMUND' o 'UNIFAC_LLE' (predictivos por contribución de "
                         "grupos), 'IDEAL' (ley de Raoult).")
    m = model.strip().upper().replace("-", "_").replace(" ", "_")
    m = {"PENG_ROBINSON": "PR", "PR78": "PR", "SOAVE": "SRK", "RKS": "SRK", "MODIFIED_UNIFAC": "UNIFAC_DORTMUND",
         "UNIFAC_DO": "UNIFAC_DORTMUND", "DORTMUND": "UNIFAC_DORTMUND", "RAOULT": "IDEAL"}.get(m, m)
    if m not in MODELS:
        raise ValueError(f"Modelo termodinámico no soportado: «{model}». Opciones: {', '.join(MODELS)}.")
    return m


def _components(components, mole_fractions) -> tuple[list[str], list[float], list[str], list[float]]:
    if not components or not isinstance(components, (list, tuple)):
        raise ValueError("Falta 'components': la lista de compuestos de la mezcla (nombre, CAS o fórmula).")
    if mole_fractions is None or len(mole_fractions) != len(components):
        raise ValueError("Falta 'mole_fractions' o no tiene un valor por componente: es la composición global de la mezcla "
                         "que se reparte entre las fases.")
    zs = [float(z) for z in mole_fractions]
    if any((not math.isfinite(z)) or z < 0 for z in zs) or sum(zs) <= 0:
        raise ValueError("mole_fractions debe contener fracciones molares no negativas con suma positiva.")
    if abs(sum(zs) - 1.0) > 1e-6:
        raise ValueError(f"mole_fractions suma {sum(zs):.8g}, no 1: no se normaliza en silencio porque una suma distinta de "
                         "1 suele indicar un componente olvidado o unidades equivocadas (fracciones másicas, porcentajes).")
    names = [str(c) for c in components]
    if len(set(n.lower() for n in names)) != len(names):
        raise ValueError("Hay componentes repetidos en 'components'.")
    keep = [i for i, z in enumerate(zs) if z > 0.0]
    if len(keep) < 2:
        raise ValueError("El flash de mezcla necesita al menos dos componentes con fracción molar positiva; para un "
                         "compuesto puro usa compound_properties (presión de vapor y fase a T y P).")
    total = sum(zs[i] for i in keep)
    return names, zs, [names[i] for i in keep], [zs[i] / total for i in keep]


def _spec(T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK) -> tuple[str, dict]:
    given = {k: v for k, v in (("T", T_K), ("P", P_Pa), ("VF", vapor_fraction), ("H", H_J_mol), ("S", S_J_molK))
             if v is not None}
    for k, v in given.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"La especificación {k} no es un número finito.")
    kind = "".join(k for k in ("T", "P", "VF", "H", "S") if k in given)
    if kind not in ("TP", "PH", "PS", "PVF", "TVF"):
        raise ValueError("El flash necesita exactamente dos especificaciones de entre: T_K y P_Pa (flash TP); P_Pa y "
                         "H_J_mol (PH); P_Pa y S_J_molK (PS); P_Pa y vapor_fraction; T_K y vapor_fraction. Se recibieron: "
                         f"{', '.join(given) if given else 'ninguna'}. Ninguna tiene valor por defecto.")
    if "T" in given and given["T"] <= 0:
        raise ValueError("T_K debe ser una temperatura absoluta positiva en kelvin.")
    if "P" in given and given["P"] <= 0:
        raise ValueError("P_Pa debe ser una presión absoluta positiva en pascales.")
    if "VF" in given and not 0.0 <= given["VF"] <= 1.0:
        raise ValueError("vapor_fraction debe estar entre 0 (punto de burbuja) y 1 (punto de rocío).")
    return kind, given


def _pair_index(names: list[str], cas: list[str], entry: dict, what: str) -> tuple[int, int]:
    if not isinstance(entry, dict) or "i" not in entry or "j" not in entry:
        raise ValueError(f"Cada elemento de {what} debe ser un diccionario con 'i' y 'j' (nombres de los dos componentes).")
    idx = []
    for key in ("i", "j"):
        q = str(entry[key]).strip().lower()
        hit = [k for k, (n, c) in enumerate(zip(names, cas)) if q in (n.strip().lower(), c)]
        if not hit:
            try:
                c2 = identify(str(entry[key]))[0].CASs
                hit = [k for k, c in enumerate(cas) if c == c2]
            except ValueError:
                hit = []
        if not hit:
            raise ValueError(f"{what}: el componente «{entry[key]}» no está entre los componentes con fracción positiva "
                             f"({', '.join(names)}).")
        idx.append(hit[0])
    if idx[0] == idx[1]:
        raise ValueError(f"{what}: un par binario necesita dos componentes distintos.")
    return idx[0], idx[1]


# ---- model construction -----------------------------------------------------------------------------------------------
def _kij_matrix(model: str, names, cas, kij, use_library_kij: bool,
                warnings: list[str]) -> tuple[list[list[float]], list[dict]]:
    from thermo.interaction_parameters import IPDB
    n = len(names)
    mat = [[0.0] * n for _ in range(n)]
    src: dict[tuple[int, int], dict] = {}
    for e in kij or []:
        i, j = _pair_index(names, cas, e, "kij")
        if "kij" not in e or not isinstance(e["kij"], (int, float)) or not math.isfinite(e["kij"]):
            raise ValueError("Cada elemento de kij necesita el valor numérico 'kij' del par.")
        a, b = min(i, j), max(i, j)
        if (a, b) in src:
            raise ValueError(f"kij: el par {names[a]}-{names[b]} está repetido.")
        mat[a][b] = mat[b][a] = float(e["kij"])
        src[(a, b)] = {"i": names[a], "j": names[b], "kij": float(e["kij"]), "source": "user"}
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) in src:
                continue
            if model == "PR" and use_library_kij and IPDB.has_ip_specific("ChemSep PR", [cas[a], cas[b]], "kij"):
                v = float(IPDB.get_ip_specific("ChemSep PR", [cas[a], cas[b]], "kij"))
                mat[a][b] = mat[b][a] = v
                src[(a, b)] = {"i": names[a], "j": names[b], "kij": v, "source": "library",
                               "library_table": "ChemSep PR (thermo.interaction_parameters.IPDB)"}
            else:
                why = ("thermo no trae tabla de kij para SRK" if model == "SRK" else
                       "use_library_kij=False" if not use_library_kij else "el par no está en la tabla ChemSep PR de thermo")
                src[(a, b)] = {"i": names[a], "j": names[b], "kij": 0.0, "source": "zero_by_absence", "reason": why}
                warnings.append(f"kij({names[a]}, {names[b]}) = 0 por ausencia de dato ({why}): no es un valor medido. Con "
                                f"kij = 0 {model} suele fallar en mezclas polares, con agua, CO2, H2S o de tamaños muy "
                                "distintos; da el kij en el argumento kij si lo tienes.")
    return mat, [src[k] for k in sorted(src)]


def _activity_matrices(model: str, names, cas, params, warnings: list[str]):
    n = len(names)
    a = [[0.0] * n for _ in range(n)]
    b = [[0.0] * n for _ in range(n)]
    al = [[0.0] * n for _ in range(n)]
    if not params:
        raise ValueError(f"El modelo {model} necesita parámetros binarios y no se dieron: pasa activity_parameters = "
                         "[{'i': "
                         "comp1, 'j': comp2, 'b_ij_K': ..., 'b_ji_K': ..., (opcionales 'a_ij', 'a_ji'), 'alpha' (solo "
                         "NRTL)}] con tau_ij = a_ij + b_ij/T. No se toman de ninguna base de datos; si no los tienes, "
                         "ajústalos a datos con la herramienta de regresión de ELV o usa un modelo UNIFAC (predictivo).")
    seen: dict[tuple[int, int], dict] = {}
    for e in params:
        i, j = _pair_index(names, cas, e, "activity_parameters")
        for k in ("b_ij_K", "b_ji_K"):
            if k not in e or not isinstance(e[k], (int, float)) or not math.isfinite(e[k]):
                raise ValueError(f"activity_parameters ({names[i]}-{names[j]}): falta '{k}' (parámetro de energía en "
                                 "kelvin, "
                                 "tau_ij = a_ij + b_ij/T).")
        if model == "NRTL" and ("alpha" not in e or not isinstance(e["alpha"], (int, float))):
            raise ValueError(f"activity_parameters ({names[i]}-{names[j]}): NRTL necesita 'alpha' (no aleatoriedad); no hay "
                             "valor por defecto.")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"activity_parameters: el par {names[key[0]]}-{names[key[1]]} está repetido.")
        a[i][j], a[j][i] = float(e.get("a_ij", 0.0)), float(e.get("a_ji", 0.0))
        b[i][j], b[j][i] = float(e["b_ij_K"]), float(e["b_ji_K"])
        if model == "NRTL":
            al[i][j] = al[j][i] = float(e["alpha"])
        seen[key] = dict({"i": names[i], "j": names[j], "a_ij": a[i][j], "a_ji": a[j][i], "b_ij_K": b[i][j],
                          "b_ji_K": b[j][i], "source": "user"}, **({"alpha": al[i][j]} if model == "NRTL" else {}))
    for x in range(n):
        for y in range(x + 1, n):
            if (x, y) not in seen:
                seen[(x, y)] = {"i": names[x], "j": names[y], "a_ij": 0.0, "a_ji": 0.0, "b_ij_K": 0.0, "b_ji_K": 0.0,
                                "source": "zero_by_absence"}
                warnings.append(f"Sin parámetros {model} para el par {names[x]}-{names[y]}: se toma tau = 0 (par ideal) por "
                                "ausencia de dato; el resultado no describe la no idealidad de ese par.")
    return a, b, al, [seen[k] for k in sorted(seen)]


def _unifac_groups(model: str, names, cas):
    from thermo.unifac import LLEUFSG, UFSG, UNIFAC_group_assignment_DDBST
    table = "MODIFIED_UNIFAC" if model == "UNIFAC_DORTMUND" else "UNIFAC"
    groups = []
    for nm, c in zip(names, cas):
        g = UNIFAC_group_assignment_DDBST(c, table)
        if not g:
            raise ValueError(f"{nm}: thermo no tiene asignación de grupos {table} (DDBST) para este compuesto; "
                             f"{model} no se "
                             "puede usar con él. Usa una ecuación de estado o un modelo con parámetros binarios propios.")
        groups.append(dict(g))
    if model == "UNIFAC_LLE":       # the LLE table (Magnussen 1981) numbers its subgroups differently: map by group name
        by_name = {}
        for k, v in LLEUFSG.items():
            by_name.setdefault(v.group, k)
        mapped = []
        for nm, g in zip(names, groups):
            m = {}
            for k, cnt in g.items():
                gname = UFSG[k].group
                if gname not in by_name:
                    raise ValueError(f"{nm}: el grupo {gname} no existe en la tabla UNIFAC-LLE de thermo; usa 'UNIFAC'.")
                m[by_name[gname]] = cnt
            mapped.append(m)
        groups = mapped
    return groups


def _build(model: str, names, cas, zs, kij, use_library_kij, activity_parameters, warnings, assumptions) -> dict:
    """thermo phases and flasher for the model, plus the provenance of every parameter."""
    from chemicals import acentric, critical
    from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVLN, GibbsExcessLiquid, IdealGas
    constants, corr = ChemicalConstantsPackage.from_IDs(cas)
    n = len(names)
    out = {"constants": constants, "corr": corr, "eos": model in MODELS_EOS}
    pure = []
    for i, (nm, c) in enumerate(zip(names, cas)):
        row = {"component": nm, "CAS": c, "molar_mass_kg_mol": constants.MWs[i] / 1000.0}
        for key, val, mf in (("Tc_K", constants.Tcs[i], critical.Tc_methods),
                             ("Pc_Pa", constants.Pcs[i], critical.Pc_methods),
                             ("acentric_factor", constants.omegas[i], acentric.omega_methods)):
            ms = mf(c)
            row[key] = val
            row[key + "_method"] = ms[0] if ms else None
            if model in MODELS_EOS:
                if val is None:
                    raise ValueError(f"{nm}: la base de datos no tiene {key}, imprescindible para la ecuación {model}. Usa "
                                     "otro modelo o comprueba el compuesto.")
                if ms and _basis_const(ms[0]) == "estimate":
                    warnings.append(f"{nm}: {key} es una ESTIMACIÓN ({ms[0]}); la ecuación {model} hereda ese error.")
        pure.append(row)
    out["pure"] = pure
    T0, P0 = 298.15, 101325.0
    if model in MODELS_EOS:
        from thermo import PRMIX, SRKMIX
        mat, table = _kij_matrix(model, names, cas, kij, use_library_kij, warnings)
        eos_kwargs = {"Tcs": constants.Tcs, "Pcs": constants.Pcs, "omegas": constants.omegas, "kijs": mat}
        cls = PRMIX if model == "PR" else SRKMIX
        gas = CEOSGas(cls, eos_kwargs, HeatCapacityGases=corr.HeatCapacityGases, T=T0, P=P0, zs=zs)
        liq = CEOSLiquid(cls, eos_kwargs, HeatCapacityGases=corr.HeatCapacityGases, T=T0, P=P0, zs=zs)
        out.update(gas=gas, liquid=liq, binary_parameters=table,
                   model_description=f"{'Peng-Robinson (1976)' if model == 'PR' else 'Soave-Redlich-Kwong (1972)'} para "
                                     "todas las fases (thermo.eos_mix), mezcla de van der Waals de un fluido con kij "
                                     "simétrico e independiente de T; sin traslación de volumen")
        assumptions.append("Entalpía y entropía: gas ideal (Cp de thermo.HeatCapacityGas de cada componente) más la función "
                           f"de desviación de la ecuación {model}.")
        assumptions.append(f"La densidad de líquido de {model} sin traslación de volumen puede errar del orden del 5-15 %; "
                           "no la uses para dimensionar.")
    else:
        if kij:
            raise ValueError(f"kij solo aplica a 'PR' y 'SRK'; con {model} los parámetros binarios van en "
                             "activity_parameters (o no hay, en los UNIFAC).")
        if model == "IDEAL":
            if activity_parameters:
                raise ValueError("El modelo 'IDEAL' no admite activity_parameters: es la ley de Raoult (γ = 1).")
            from thermo.activity import IdealSolution
            ge = IdealSolution(T=T0, xs=zs)
            table = []
            desc = "ley de Raoult (disolución ideal, γ_i = 1), thermo.activity.IdealSolution"
            warnings.append("Modelo IDEAL (ley de Raoult): no describe azeótropos ni dos fases líquidas; solo es razonable "
                            "para mezclas de compuestos muy parecidos a baja presión.")
        elif model in MODELS_GE:
            a, b, al, table = _activity_matrices(model, names, cas, activity_parameters, warnings)
            if model == "NRTL":
                from thermo.nrtl import NRTL
                ge = NRTL(T=T0, xs=zs, tau_as=a, tau_bs=b, alpha_cs=al)
                desc = "NRTL (Renon y Prausnitz 1968), tau_ij = a_ij + b_ij/T, thermo.nrtl.NRTL"
            elif model == "UNIQUAC":
                from thermo.unifac import UNIFAC_RQ
                from thermo.uniquac import UNIQUAC
                rq = [UNIFAC_RQ(g) for g in _unifac_groups("UNIFAC", names, cas)]
                ge = UNIQUAC(T=T0, xs=zs, rs=[float(r) for r, _ in rq], qs=[float(q) for _, q in rq], tau_as=a, tau_bs=b)
                desc = "UNIQUAC (Abrams y Prausnitz 1975), tau_ij = exp(a_ij + b_ij/T), z = 10, thermo.uniquac.UNIQUAC"
                for row, (r, q) in zip(pure, rq):
                    row.update(uniquac_r=float(r), uniquac_q=float(q))
                assumptions.append("UNIQUAC: r y q de cada componente son la suma de Rk y Qk de sus subgrupos UNIFAC "
                                   "(asignación DDBST de thermo), como en la herramienta de regresión de ELV; si tus "
                                   "parámetros se ajustaron con otros r y q, el resultado no es el de tu ajuste.")
            else:
                from thermo.wilson import Wilson
                vols = []
                for row, vl in zip(pure, corr.VolumeLiquids):
                    v = vl.T_dependent_property(WILSON_VOLUME_T)
                    if v is None:
                        raise ValueError(f"{row['component']}: thermo no da volumen molar de líquido a 298.15 K, necesario "
                                         "para Wilson.")
                    vols.append(float(v))
                    row.update(wilson_V_m3_mol=float(v), wilson_V_method=vl.method)
                la = [[(a[i][j] + math.log(vols[j] / vols[i])) if i != j else 0.0 for j in range(n)] for i in range(n)]
                ge = Wilson(T=T0, xs=zs, lambda_as=la, lambda_bs=b)
                desc = "Wilson (1964), Λ_ij = (V_j/V_i)·exp(a_ij + b_ij/T), thermo.wilson.Wilson"
                assumptions.append("Wilson: volúmenes molares de líquido de thermo.VolumeLiquid a 298.15 K (como en la "
                                   "herramienta de regresión de ELV). Wilson no puede representar dos fases líquidas: con "
                                   "él nunca aparecerá ELL.")
            if activity_parameters and any(e.get("a_ij") is None and e.get("a_ji") is None for e in activity_parameters):
                assumptions.append("Parámetros de actividad sin a_ij: se toma a_ij = 0 (forma clásica con energía "
                                   "independiente de T).")
            assumptions.append("Los parámetros de actividad van ligados a la correlación de presión de vapor con la que se "
                               "ajustaron: aquí se usa la de thermo, así que un ajuste hecho con otra Antoine no se "
                               "reproduce exactamente (décimas de kelvin en temperaturas de burbuja).")
        else:
            from thermo.unifac import DOUFIP2016, DOUFSG, LLEUFIP, LLEUFSG, UFIP, UFSG, UNIFAC
            groups = _unifac_groups(model, names, cas)
            sub, ip, ver, ref = {"UNIFAC": (UFSG, UFIP, 0, "UNIFAC original de ELV (Fredenslund 1975; tabla de Hansen 1991 "
                                                           "y posteriores)"),
                                 "UNIFAC_DORTMUND": (DOUFSG, DOUFIP2016, 1, "UNIFAC modificado Dortmund (Gmehling; "
                                                                            "parámetros publicados hasta 2016)"),
                                 "UNIFAC_LLE": (LLEUFSG, LLEUFIP, 0,
                                                "UNIFAC-LLE (Magnussen, Rasmussen y Fredenslund 1981)")}[model]
            ge = UNIFAC.from_subgroups(T=T0, xs=zs, chemgroups=groups, version=ver, interaction_data=ip, subgroups=sub)
            desc = f"{ref}, thermo.unifac.UNIFAC; modelo PREDICTIVO, sin parámetros ajustados a esta mezcla"
            table = []
            for row, g in zip(pure, groups):
                row["unifac_subgroups"] = {sub[k].group: cnt for k, cnt in g.items()}
            warnings.append(f"{model} es una predicción por contribución de grupos, no un ajuste a datos de esta mezcla: "
                            "errores típicos de varios % en presión de burbuja y mayores en solubilidades mutuas.")
        liq = GibbsExcessLiquid(VaporPressures=corr.VaporPressures, HeatCapacityGases=corr.HeatCapacityGases,
                                VolumeLiquids=corr.VolumeLiquids, GibbsExcessModel=ge,
                                EnthalpyVaporizations=corr.EnthalpyVaporizations, equilibrium_basis="Psat",
                                caloric_basis="Psat", T=T0, P=P0, zs=zs)
        gas = IdealGas(HeatCapacityGases=corr.HeatCapacityGases, T=T0, P=P0, zs=zs)
        out.update(gas=gas, liquid=liq, binary_parameters=table,
                   model_description=desc + "; fase vapor gas ideal; equilibrio γ_i·x_i·Psat_i = y_i·P sin corrección de "
                                            "Poynting ni φ de saturación")
        for row, vp in zip(pure, corr.VaporPressures):
            row.update(Psat_method=vp.method,
                       Psat_T_range_K=[float(x) for x in vp.T_limits[vp.method]] if vp.method else None)
            if vp.method and _basis_tdep(vp.method) != "data":
                warnings.append(f"{row['component']}: la presión de vapor es una ESTIMACIÓN ({vp.method}).")
        assumptions.append("Presión de vapor de cada componente: método por defecto de thermo.VaporPressure (indicado en "
                           "pure_component_data); fuera de su intervalo se extrapola y se avisa.")
        assumptions.append("Entalpía del líquido: gas ideal menos ΔHvap(T) de cada componente (thermo.EnthalpyVaporization) "
                           "más la entalpía de exceso del modelo; la densidad del líquido es la mezcla ideal de volúmenes "
                           "molares.")
        assumptions.append("Válido a presión baja o moderada (gas ideal): por encima de unos 5-10 bar, o con componentes "
                           "supercríticos, usa una ecuación de estado.")
    out["flasher"] = FlashVLN(constants, corr, liquids=[out["liquid"], out["liquid"]], gas=out["gas"])
    return out


# ---- independent checks ------------------------------------------------------------------------------------------------
def rachford_rice(mole_fractions: list[float], K_values: list[float]) -> dict:
    """Two-phase split for FIXED K-values: Σ z_i (K_i − 1)/(1 + β (K_i − 1)) = 0 solved for β (fraction of the phase in
    the numerator of K) by Brent between the asymptotes 1/(1 − K_max) and 1/(1 − K_min) (Whitson and Michelsen 1989).
    β outside [0, 1] means the feed is single-phase with these K-values (negative flash); it is returned as computed."""
    zs, Ks = [float(z) for z in mole_fractions], [float(k) for k in K_values]
    if len(zs) != len(Ks) or len(zs) < 2:
        raise ValueError("Rachford-Rice necesita una K por componente y al menos dos componentes.")
    if any(k <= 0 for k in Ks) or any(z < 0 for z in zs) or abs(sum(zs) - 1.0) > 1e-9:
        raise ValueError("Rachford-Rice necesita K positivas y fracciones molares no negativas que sumen 1.")
    kmax, kmin = max(Ks), min(Ks)
    if kmax <= 1.0 or kmin >= 1.0:
        raise ValueError("Con todas las K a un mismo lado de 1 no hay solución de dos fases (Rachford-Rice no tiene raíz).")

    def f(beta: float) -> float:
        return sum(z * (k - 1.0) / (1.0 + beta * (k - 1.0)) for z, k in zip(zs, Ks))

    lo, hi = 1.0 / (1.0 - kmax), 1.0 / (1.0 - kmin)
    eps = 1e-12 * (hi - lo)
    beta = brentq(f, lo + eps, hi - eps, xtol=1e-15, rtol=1e-14, maxiter=200)
    xs = [z / (1.0 + beta * (k - 1.0)) for z, k in zip(zs, Ks)]
    return {"phase_fraction": beta, "x": xs, "y": [k * x for k, x in zip(Ks, xs)], "residual": f(beta)}



def _lnphis(m: dict, T: float, P: float, w: list[float], kind: str) -> tuple[list[float], str]:
    """ln φ_i of a phase of composition w and what it is ("gas" or "liquid"). EOS: the root of lower Gibbs energy
    (a single root is named by thermo's phase identification parameter, PIP > 1 = liquid); γ-φ: `kind` picks the model."""
    if m["eos"]:
        found = []
        for name in ("gas", "liquid"):
            try:
                ph = m[name].to(T=T, P=P, zs=w)
                lp = [float(x) for x in ph.lnphis()]
            except Exception:  # noqa: BLE001 — a root may not exist at this composition
                continue
            found.append((sum(wi * li for wi, li in zip(w, lp)), lp, name, ph))
        if not found:
            raise ArithmeticError("la ecuación de estado no da raíz para la composición de prueba")
        if len(found) == 2 and abs(found[0][3].Z() - found[1][3].Z()) < 1e-10:        # one real root
            return found[0][1], ("liquid" if found[0][3].PIP() > 1.0 else "gas")
        best = min(found, key=lambda f: f[0])
        return best[1], best[2]
    if kind == "gas":
        return [0.0] * len(w), "gas"
    return [float(x) for x in m["liquid"].to(T=T, P=P, zs=w).lnphis()], "liquid"


def _wilson_K(m: dict, T: float, P: float) -> list[float] | None:
    c = m["constants"]
    if any(x is None for x in list(c.Tcs) + list(c.Pcs) + list(c.omegas)):
        return None
    return [pc / P * math.exp(5.373 * (1.0 + om) * (1.0 - tc / T)) for tc, pc, om in zip(c.Tcs, c.Pcs, c.omegas)]


def _tpd(m: dict, T: float, P: float, d: list[float], phases: list[list[float]], z: list[float],
         tol: float = TPD_TOL) -> dict:
    """Michelsen tangent-plane test by successive substitution. d_i = ln x_i + ln φ_i of the reference phase."""
    n = len(z)
    starts: list[tuple[str, list[float], str]] = []
    K = _wilson_K(m, T, P)
    if K:
        starts.append(("wilson_vapor", [zi * ki for zi, ki in zip(z, K)], "gas" if not m["eos"] else "any"))
        starts.append(("wilson_liquid", [zi / ki for zi, ki in zip(z, K)], "liquid"))
    for i in range(n):
        starts.append((f"pure_{i}", [(1.0 - 1e-6) if j == i else 1e-6 / (n - 1) for j in range(n)], "liquid"))
    if not m["eos"]:
        starts.append(("ideal_gas", list(z), "gas"))
    trials = []
    for label, W, kind in starts:
        s = sum(W)
        W = [max(x / s, 1e-300) for x in W]
        if not m["eos"] and kind == "gas":                    # ideal gas: ln φ = 0, the stationary point is explicit
            W = [math.exp(di) for di in d]
            it, converged, root = 1, True, "gas"
        else:
            converged, it, root = False, 0, kind
            try:
                for it in range(1, TPD_MAXITER + 1):
                    s = sum(W)
                    lp, root = _lnphis(m, T, P, [x / s for x in W], kind)
                    Wn = [math.exp(min(di - li, 600.0)) for di, li in zip(d, lp)]
                    err = max(abs(math.log(a) - math.log(b)) for a, b in zip(Wn, W) if a > 0 and b > 0)
                    W = [max(x, 1e-300) for x in Wn]
                    if err < TPD_XTOL:
                        converged = True
                        break
            except (ArithmeticError, ValueError, OverflowError) as e:
                trials.append({"start": label, "converged": False, "error": str(e)[:120]})
                continue
        s = sum(W)
        w = [x / s for x in W]
        try:
            lp, root = _lnphis(m, T, P, w, kind if not (not m["eos"] and kind == "gas") else "gas")
        except (ArithmeticError, ValueError, OverflowError) as e:
            trials.append({"start": label, "converged": False, "error": str(e)[:120]})
            continue
        tm = 1.0 + sum(Wi * (math.log(Wi) + li - di - 1.0) for Wi, li, di in zip(W, lp, d))
        dist = min(max(abs(a - b) for a, b in zip(w, x)) for x in phases)
        trials.append({"start": label, "converged": converged, "iterations": it, "tm": tm, "root": root,
                       "trivial": dist < TRIVIAL_TOL, "distance_to_nearest_phase": dist, "composition": w})
    bad = [t for t in trials if "tm" in t and t["tm"] < -tol and not t["trivial"]]
    worst = min(bad, key=lambda t: t["tm"]) if bad else None
    return {"stable": not bad, "n_trials": len(trials), "n_converged": sum(1 for t in trials if t.get("converged")),
            "n_failed": sum(1 for t in trials if "error" in t),
            "min_tm": min((t["tm"] for t in trials if "tm" in t), default=None),
            "unstable_trial": ({"start": worst["start"], "tm": worst["tm"], "root": worst["root"],
                                "mole_fractions": worst["composition"]} if worst else None)}


class _State:
    """An equilibrium state: T, P and the phases as (is_gas, fraction, composition, thermo phase object)."""

    def __init__(self, T: float, P: float, entries: list[tuple], solver: str):
        self.T, self.P, self.entries, self.solver = float(T), float(P), entries, solver

    def H(self) -> float:
        return sum(b * float(ph.H()) for _, b, _, ph in self.entries)

    def S(self) -> float:
        return sum(b * float(ph.S()) for _, b, _, ph in self.entries)


def _state_from_thermo(res) -> _State:
    entries = [(res.gas is not None and ph is res.gas, float(b), [float(x) for x in ph.zs], ph)
               for ph, b in zip(res.phases, res.betas)]
    return _State(res.T, res.P, entries, "thermo.FlashVLN")


def _min_q(zs: list[float], inv_phi: list[list[float]], betas: list[float]) -> list[float]:
    """Phase fractions for fixed fugacity coefficients: minimum of Michelsen's convex Q(β) = Σβ_k − Σ z_i ln E_i,
    E_i = Σ_k β_k/φ_ik, subject to β_k ≥ 0 (active-set Newton). A phase that ends at β = 0 is absent."""
    A, z, b = np.array(inv_phi, float), np.array(zs, float), np.array(betas, float)
    F = len(b)

    def q(v):
        E = v @ A
        return float(v.sum() - (z * np.log(E)).sum()) if (E > 0).all() else math.inf

    for _ in range(200):
        E = b @ A
        g = 1.0 - A @ (z / E)
        H = (A * (z / E ** 2)) @ A.T
        free = [k for k in range(F) if b[k] > 0.0 or g[k] < 0.0]
        d = np.zeros(F)
        while free:
            sol = np.linalg.lstsq(H[np.ix_(free, free)], -g[free], rcond=None)[0]
            blocked = [k for k, dk in zip(free, sol) if b[k] <= 0.0 and dk < 0.0]
            if not blocked:
                d[free] = sol
                break
            free = [k for k in free if k not in blocked]
        if not free or max(abs(g[k]) for k in free) < 1e-14:
            break
        alpha = 1.0
        for k in free:
            if d[k] < 0.0:
                alpha = min(alpha, -b[k] / d[k])
        q0 = q(b)
        while alpha > 1e-14 and q(np.maximum(b + alpha * d, 0.0)) > q0 + 1e-15 * abs(q0):
            alpha *= 0.5
        if alpha <= 1e-14:
            break
        b = np.maximum(b + alpha * d, 0.0)
    return [float(v) for v in b]


def _multiphase_ss(m: dict, T: float, P: float, zs: list[float], comps: list[list[float]], kinds: list[str]):
    """Multiphase successive substitution (Michelsen): fugacity coefficients at the current compositions, phase
    fractions by `_min_q`, new compositions x_ik = z_i/(E_i φ_ik). Returns [(root, fraction, composition)] or None."""
    n = len(zs)
    comps = [[max(x, 1e-300) / sum(c) for x in c] for c in comps]
    betas = [1.0 / len(comps)] * len(comps)
    roots = list(kinds)
    idle = [0] * len(comps)
    converged = False
    for _ in range(SS_MAXITER):
        inv = []
        for k, (x, kind) in enumerate(zip(comps, kinds)):
            lp, roots[k] = _lnphis(m, T, P, x, kind)
            inv.append([math.exp(-min(max(v, -600.0), 600.0)) for v in lp])
        betas = _min_q(zs, inv, betas)
        E = [sum(betas[k] * inv[k][i] for k in range(len(comps))) for i in range(n)]
        err = 0.0
        new = []
        for k in range(len(comps)):
            row = [zs[i] * inv[k][i] / E[i] for i in range(n)]
            tot = sum(row)
            row = [max(v / tot, 1e-300) for v in row]
            if betas[k] > 0.0:
                err = max(err, max(abs(math.log(a) - math.log(b)) for a, b in zip(row, comps[k])))
                idle[k] = 0
            else:
                idle[k] += 1
            new.append(row)
        comps = new
        keep = [k for k in range(len(comps)) if idle[k] < 25]
        for a in list(keep):                                   # merge phases that collapsed onto each other
            for b in list(keep):
                if a < b and a in keep and b in keep and roots[a] == roots[b] \
                        and max(abs(u - v) for u, v in zip(comps[a], comps[b])) < 1e-9:
                    betas[a] += betas[b]
                    keep.remove(b)
        if len(keep) < len(comps):
            comps, kinds, roots = [comps[k] for k in keep], [kinds[k] for k in keep], [roots[k] for k in keep]
            betas, idle = [betas[k] for k in keep], [idle[k] for k in keep]
            tot = sum(betas)
            betas = [v / tot for v in betas] if tot > 0 else [1.0 / len(keep)] * len(keep)
            continue
        if err < SS_XTOL:
            converged = True
            break
    active = [k for k in range(len(comps)) if betas[k] > 1e-13]
    if not converged or not active:                          # never return an unconverged state as an equilibrium
        return None
    tot = sum(betas[k] for k in active)
    return [(roots[k], betas[k] / tot, comps[k]) for k in active]


def _fug_residual(entries: list[tuple]) -> tuple[float, list[float]]:
    lnf = [[math.log(max(x, 1e-300)) + float(lp) for x, lp in zip(xs, ph.lnphis())] for _, _, xs, ph in entries]
    return max((abs(a - b) for row in lnf[1:] for a, b in zip(row, lnf[0])), default=0.0), lnf[0]


def _tp_state(m: dict, T: float, P: float, zs: list[float], robust: bool = True) -> _State:
    """T-P flash: thermo first; if the independent tangent-plane test finds a missing phase, the state is re-solved
    with the multiphase successive substitution above, seeded with the unstable trial (same model, other solver)."""
    state = _state_from_thermo(m["flasher"].flash(T=T, P=P, zs=zs))
    if not robust:
        return state
    for _ in range(3):
        res, d = _fug_residual(state.entries)
        st = _tpd(m, T, P, d, [e[2] for e in state.entries], zs, max(TPD_TOL, 10.0 * res))
        if st["stable"] is not False:
            break
        trial = st["unstable_trial"]
        w = trial["mole_fractions"]
        seeds = [(e[2], ("any" if m["eos"] else ("gas" if e[0] else "liquid"))) for e in state.entries]
        seeds.sort(key=lambda sd: max(abs(a - b) for a, b in zip(sd[0], w)))
        new = (w, "any" if m["eos"] else trial["root"])
        # candidate phase sets: add the trial phase (if the phase rule allows another phase at fixed T, P), or replace one
        # of the current phases by it, closest first
        cands = ([seeds + [new]] if len(seeds) < min(3, len(zs)) else []) + \
            [seeds[:k] + seeds[k + 1:] + [new] for k in range(len(seeds))]
        fallback = None
        for cand in cands:
            try:
                sol = _multiphase_ss(m, T, P, zs, [sd[0] for sd in cand], [sd[1] for sd in cand])
            except (ArithmeticError, ValueError, OverflowError, np.linalg.LinAlgError):
                sol = None
            if not sol:
                continue
            entries = [(root == "gas", beta, xs, m[root].to(T=T, P=P, zs=xs)) for root, beta, xs in sol]
            cand_state = _State(T, P, entries, "own_multiphase_successive_substitution")
            fallback = fallback or cand_state
            if _state_is_valid(m, cand_state, zs):
                fallback = cand_state
                break
        if fallback is None:
            break
        state = fallback
    return state


def _state_is_valid(m: dict, state: _State, zs: list[float]) -> bool:
    """Material balance closes and the independent tangent-plane test finds no missing phase."""
    mb = max(abs(sum(b * x[i] for _, b, x, _ in state.entries) - zs[i]) for i in range(len(zs)))
    if mb > BALANCE_TOL:
        return False
    res, d = _fug_residual(state.entries)
    if res > FUGACITY_TOL:
        return False
    return _tpd(m, state.T, state.P, d, [e[2] for e in state.entries], zs, max(TPD_TOL, 10.0 * res))["stable"] is not False


def _solve_T(m: dict, P: float, zs: list[float], prop: str, target: float, notes: list[str], robust: bool = False,
             around: float | None = None) -> _State:
    """P-H / P-S: bracketed root in T over T-P flashes (monotonic at fixed P; isothermal jumps are reported as residual)."""
    def f(T):
        st = _tp_state(m, T, P, zs, robust)
        return (st.H() if prop == "H" else st.S()) - target

    step, nmax = T_BRACKET
    T0 = around if around is not None else 298.15
    lo = hi = T0
    flo = fhi = f(T0)
    k = 0
    while flo > 0 and k < nmax:
        lo /= step
        flo = f(lo)
        k += 1
    k = 0
    while fhi < 0 and k < nmax:
        hi *= step
        fhi = f(hi)
        k += 1
    if flo > 0 or fhi < 0:
        raise ArithmeticError(f"no se pudo acotar la temperatura entre {lo:.1f} y {hi:.1f} K para la {prop} pedida")
    T = T0 if lo == hi else brentq(f, lo, hi, xtol=1e-7, rtol=1e-13, maxiter=200)
    state = _tp_state(m, T, P, zs, True)
    if state.solver != "thermo.FlashVLN" and not robust:       # the stability test changed the phases: search again with it
        return _solve_T(m, P, zs, prop, target, notes, True, T)
    notes.append(f"Flash P-{prop}: búsqueda de raíz acotada (Brent) en T sobre flashes T-P, entre {lo:.1f} y {hi:.1f} K.")
    if abs(_spec_value(state, prop) - target) > SPEC_TOL * _spec_scale(prop, target, T):
        dT = max(JUMP_REL * T, 1e-6)
        jump = _resolve_jump(m, T, P, zs, [_tp_state(m, T - dT, P, zs), state, _tp_state(m, T + dT, P, zs)], prop, target)
        if jump is not None:
            notes.append(_JUMP_NOTE.format(var="T", x=f"{T:.6f} K"))
            return jump
    return state


JUMP_REL = 1e-8             # relative offset of T (or P) on each side of an isothermal / isobaric jump to collect its phases
VF_XTOL_REL = 1e-10         # relative width of the final bracket of the P-VF / T-VF search
_JUMP_NOTE = ("La especificación cae en un salto de la función a {var} = {x} (equilibrio trifásico de una mezcla "
              "binaria, heteroazeótropo): las fases de los dos lados del salto se reunieron en un estado trifásico y sus "
              "fracciones se resolvieron del balance de materia y de la especificación (sistema lineal con fracciones no "
              "negativas).")


def _spec_value(state: _State, prop: str) -> float:
    if prop == "H":
        return state.H()
    if prop == "S":
        return state.S()
    return sum(b for is_gas, b, _, _ in state.entries if is_gas)


def _spec_scale(prop: str, target: float, T: float) -> float:
    return max(abs(target), R_GAS * T) if prop == "H" else max(abs(target), R_GAS) if prop == "S" else 1.0


def _resolve_jump(m: dict, T: float, P: float, zs: list[float], states: list[_State], prop: str, target: float):
    """At a discontinuity of H(T), S(T) or VF(T) (three-phase line of a binary), gather the distinct phases found on both
    sides, evaluate them at (T, P) and solve their fractions from Σβ_k x_ik = z_i and Σβ_k s_k = target, β ≥ 0 (NNLS).
    Returns a three-phase state or None when the phases or the linear system do not close."""
    from scipy.optimize import nnls
    found: list[tuple[str, list[float]]] = []
    for st in states:
        for is_gas, _, xs, _ in st.entries:
            root = "gas" if is_gas else "liquid"
            if all(r != root or max(abs(a - b) for a, b in zip(x, xs)) > 1e-5 for r, x in found):
                found.append((root, list(xs)))
    if len(found) < 3:
        return None
    try:
        objs = [m[root].to(T=T, P=P, zs=xs) for root, xs in found]
    except Exception:  # noqa: BLE001
        return None
    spec = [float(o.H()) if prop == "H" else float(o.S()) if prop == "S" else (1.0 if r == "gas" else 0.0)
            for (r, _), o in zip(found, objs)]
    scale = _spec_scale(prop, target, T)
    A = np.array([[xs[i] for _, xs in found] for i in range(len(zs))] + [[1.0] * len(found)] + [[s / scale for s in spec]])
    b = np.array(list(zs) + [1.0, target / scale])
    beta, resid = nnls(A, b)
    if resid > 1e-9:
        return None
    entries = [(r == "gas", float(bk), xs, o) for (r, xs), o, bk in zip(found, objs, beta) if bk > 1e-14 or r == "gas"]
    tot = sum(e[1] for e in entries)
    entries = [(g, bk / tot, xs, o) for g, bk, xs, o in entries]
    return _State(T, P, entries, "own_three_phase_jump_resolution")


def _solve_vf(m: dict, zs: list[float], var: str, fixed: float, target: float, x0: float | None, notes: list[str]) -> _State:
    """P-VF (var = "T") or T-VF (var = "P") by bisection over VLLE-aware T-P flashes on the predicate «vapour fraction
    above the target» (VF rises with T and falls with P); used when thermo's two-phase VF solver gives an invalid state."""
    def flash(x):
        st = _tp_state(m, x, fixed, zs) if var == "T" else _tp_state(m, fixed, x, zs)
        return st, _spec_value(st, "VF")

    def above(vf):
        hit = vf >= 1.0 - 1e-12 if target >= 1.0 else vf > target
        return hit if var == "T" else not hit

    step, nmax = T_BRACKET
    x = x0 if x0 is not None and math.isfinite(x0) and x0 > 0 else (298.15 if var == "T" else 101325.0)
    st, vf = flash(x)
    lo = hi = None
    lo_st = hi_st = None
    if above(vf):
        hi, hi_st = x, st
        for _ in range(nmax):
            x /= step
            st, vf = flash(x)
            if not above(vf):
                lo, lo_st = x, st
                break
            hi, hi_st = x, st
    else:
        lo, lo_st = x, st
        for _ in range(nmax):
            x *= step
            st, vf = flash(x)
            if above(vf):
                hi, hi_st = x, st
                break
            lo, lo_st = x, st
    if lo is None or hi is None:
        raise ArithmeticError(f"no se pudo acotar {'la temperatura' if var == 'T' else 'la presión'} para la fracción de "
                              "vapor pedida")
    while hi - lo > VF_XTOL_REL * hi:
        mid = math.sqrt(lo * hi)
        st, vf = flash(mid)
        if above(vf):
            hi, hi_st = mid, st
        else:
            lo, lo_st = mid, st
    hit_st, miss_st = (hi_st, lo_st) if var == "T" else (lo_st, hi_st)     # hit: VF above the target (or = 1, dew point)
    if target <= 0.0:
        best = hit_st                                            # bubble: the side with an incipient vapour
    elif target >= 1.0:
        best = miss_st                                           # dew: the side with an incipient liquid
    else:
        best = min((lo_st, hi_st), key=lambda s: abs(_spec_value(s, "VF") - target))
    unit = "K" if var == "T" else "Pa"
    notes.append(f"Flash {'P' if var == 'T' else 'T'}-VF: el solver de dos fases de thermo dio un estado inválido "
                 "(balance o "
                 f"estabilidad); se resolvió por bisección en {var} sobre flashes T-P con prueba de estabilidad, hasta un "
                 f"intervalo relativo de {VF_XTOL_REL:g} ({lo:.10g}-{hi:.10g} {unit}).")
    if abs(_spec_value(best, "VF") - target) > SPEC_TOL:
        T, P = (hi, fixed) if var == "T" else (fixed, hi)
        jump = _resolve_jump(m, T, P, zs, [lo_st, hi_st], "VF", target)
        if jump is not None:
            notes.append(_JUMP_NOTE.format(var=var, x=f"{hi:.8g} {unit}"))
            return jump
    return best


def _phase_table(state: _State, names: list[str], mws: list[float]) -> tuple[list[dict], list]:
    """Phases labelled V, L or L1/L2 (L1 = lighter liquid); a phase of zero fraction is the incipient one."""
    def mw(e) -> float:
        return sum(x * w for x, w in zip(e[2], mws)) / 1000.0

    liquids = sorted((e for e in state.entries if not e[0]), key=lambda e: mw(e) / float(e[3].V()))
    rows = []
    for e in state.entries:
        is_gas, beta, xs, ph = e
        label = "V" if is_gas else ("L" if len(liquids) == 1 else f"L{[id(q) for q in liquids].index(id(e)) + 1}")
        row = {"label": label, "type": "vapor" if is_gas else "liquid", "mole_fraction_of_feed": float(beta),
               "mole_fractions": {nm: x for nm, x in zip(names, xs)},
               "molar_mass_kg_mol": mw(e), "rho_kg_m3": mw(e) / float(ph.V()), "Z": float(ph.Z()),
               "H_J_mol": float(ph.H()), "S_J_molK": float(ph.S()),
               "ln_fugacity_coefficients": [float(v) for v in ph.lnphis()]}
        rows.append((row, ph, xs))
    rows.sort(key=lambda r: (r[0]["label"] != "V", r[0]["label"]))
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows]


def _evidence(m: dict, res, table: list[dict], objs: list, zs: list[float], kind: str, given: dict) -> dict:
    T, P = float(res.T), float(res.P)
    n = len(zs)
    betas = [row["mole_fraction_of_feed"] for row in table]
    xs = [x for _, x in objs]
    ev: dict = {"verified": False, "reasons": []}
    ev["material_balance_max_abs"] = max(abs(sum(b * x[i] for b, x in zip(betas, xs)) - zs[i]) for i in range(n))
    ev["phase_fractions_sum_minus_1"] = sum(betas) - 1.0
    ev["composition_sum_minus_1_max_abs"] = max(abs(sum(x) - 1.0) for x in xs)
    lnf = [[math.log(max(x[i], 1e-300)) + row["ln_fugacity_coefficients"][i] for i in range(n)] for row, x in zip(table, xs)]
    ev["fugacity_equality_max_abs_dlnf"] = max((abs(lnf[k][i] - lnf[0][i]) for k in range(1, len(xs)) for i in range(n)),
                                               default=0.0)
    rr = 0.0
    for k in range(1, len(xs)):
        Ks = [[xs[j][i] / max(xs[0][i], 1e-300) for i in range(n)] for j in range(len(xs))]
        den = [1.0 + sum(betas[j] * (Ks[j][i] - 1.0) for j in range(1, len(xs))) for i in range(n)]
        rr = max(rr, abs(sum(zs[i] * (Ks[k][i] - 1.0) / den[i] for i in range(n))))
    ev["rachford_rice_max_abs_residual"] = rr
    if len(xs) == 2 and all(x > 0 for x in xs[1]):
        try:                                   # phase fraction re-solved from the K-values of the solution (own solver)
            beta_rr = rachford_rice(zs, [xs[0][i] / xs[1][i] for i in range(n)])["phase_fraction"]
            ev["rachford_rice_phase_fraction_difference"] = min(max(beta_rr, 0.0), 1.0) - betas[0]
        except ValueError:
            ev["rachford_rice_phase_fraction_difference"] = None
    d = lnf[0]
    try:
        st = _tpd(m, T, P, d, xs, zs, max(TPD_TOL, 10.0 * ev["fugacity_equality_max_abs_dlnf"]))
    except Exception as e:  # noqa: BLE001 — the independent test must never hide the flash result
        st = {"stable": None, "error": str(e)[:200]}
    # was the feed, taken as one phase, stable? (what a flash without stability analysis would have assumed)
    feed = None
    try:
        cands = []
        for kind_ in (("any",) if m["eos"] else ("gas", "liquid")):
            lp, root = _lnphis(m, T, P, zs, kind_)
            cands.append((sum(z * (math.log(z) + li) for z, li in zip(zs, lp)), lp, root))
        g, lp, root = min(cands, key=lambda c: c[0])
        feed = _tpd(m, T, P, [math.log(z) + li for z, li in zip(zs, lp)], [zs], zs)
        feed = {"stable": feed["stable"], "min_tm": feed["min_tm"], "single_phase_root": root}
    except Exception as e:  # noqa: BLE001
        feed = {"stable": None, "error": str(e)[:200]}
    n_liq = sum(1 for row in table if row["type"] == "liquid")
    ev["stability"] = {
        "final_state_stable": st.get("stable"), "trial_phases": st.get("n_trials"),
        "trials_converged": st.get("n_converged"),
        "trials_failed": st.get("n_failed"), "min_tangent_plane_distance": st.get("min_tm"),
        "unstable_trial": st.get("unstable_trial"), "single_phase_feed": feed, "second_liquid_found": n_liq >= 2,
        "test": "Michelsen (1982), sustitución sucesiva propia desde K de Wilson (vapor y líquido) y cada componente puro; "
                "independiente de la prueba interna de thermo"}
    if "error" in st:
        ev["stability"]["error"] = st["error"]
    spec_res = None
    if kind in ("PH", "PS"):
        target = given["H"] if kind == "PH" else given["S"]
        val = float(res.H()) if kind == "PH" else float(res.S())
        scale = max(abs(target), R_GAS * T if kind == "PH" else R_GAS)
        spec_res = (val - target) / scale
    elif kind in ("PVF", "TVF"):
        vf = sum(row["mole_fraction_of_feed"] for row in table if row["type"] == "vapor")
        spec_res = vf - given["VF"]
    if spec_res is not None:
        ev["specification_residual"] = spec_res
    if ev["material_balance_max_abs"] > BALANCE_TOL or abs(ev["phase_fractions_sum_minus_1"]) > BALANCE_TOL \
            or ev["composition_sum_minus_1_max_abs"] > BALANCE_TOL:
        ev["reasons"].append("El balance de materia entre fases o la normalización de composiciones no cierra a 1e-8.")
    if ev["fugacity_equality_max_abs_dlnf"] > FUGACITY_TOL:
        ev["reasons"].append("Las fugacidades entre fases difieren (máx |Δ ln f| = "
                             f"{ev['fugacity_equality_max_abs_dlnf']:.2e} > {FUGACITY_TOL:g}): el flash no convergió a un "
                             "equilibrio.")
    if rr > RR_TOL:
        ev["reasons"].append(f"Residuo de Rachford-Rice {rr:.2e} > {RR_TOL:g}.")
    drr = ev.get("rachford_rice_phase_fraction_difference")
    if drr is not None and abs(drr) > RR_BETA_TOL:
        ev["reasons"].append(f"La fracción de fase resuelta de nuevo con Rachford-Rice difiere en {drr:.2e} de la del "
                             "flash.")
    if st.get("stable") is False:
        u = st["unstable_trial"]
        ev["reasons"].append("La prueba de estabilidad independiente encuentra una fase de prueba con distancia al plano "
                             f"tangente negativa (tm = {u['tm']:.3e}, partida {u['start']}): falta una fase en el "
                             "resultado.")
    elif st.get("stable") is None:
        ev["reasons"].append("La prueba de estabilidad independiente no pudo ejecutarse: " + st.get("error", ""))
    if spec_res is not None and abs(spec_res) > SPEC_TOL:
        ev["reasons"].append(
            f"La especificación no se cumple (residuo {spec_res:.3e}): a esta presión el valor pedido cae en un salto "
            "isotérmico (azeótropo o equilibrio trifásico univariante) o el solver no convergió; las fracciones de fase no "
            "quedan determinadas.")
    ev["tolerances"] = {"balance": BALANCE_TOL, "fugacity_dlnf": FUGACITY_TOL, "rachford_rice": RR_TOL,
                        "rachford_rice_phase_fraction": RR_BETA_TOL, "tpd": TPD_TOL,
                        "specification": SPEC_TOL}
    ev["verified"] = not ev["reasons"]
    return ev


# ---- public engines ---------------------------------------------------------------------------------------------------
def independent_flash(components: list[str], mole_fractions: list[float], model: str | None = None,
                      T_K: float | None = None, P_Pa: float | None = None, vapor_fraction: float | None = None,
                      H_J_mol: float | None = None, S_J_molK: float | None = None, kij: list[dict] | None = None,
                      activity_parameters: list[dict] | None = None, use_library_kij: bool = True) -> dict:
    """Flash of a mixture with thermo (no DWSIM): phases, fractions, compositions, K-values, H and S, with a Michelsen
    stability test that detects a second liquid. `model` and two specifications are mandatory (see module docstring)."""
    import thermo
    mdl = _require_model(model)
    all_names, all_zs, names, zs = _components(components, mole_fractions)
    kind, given = _spec(T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK)
    warnings: list[str] = []
    assumptions: list[str] = []
    metas = [identify(nm)[0] for nm in names]
    cas = [mt.CASs for mt in metas]
    if len(set(cas)) != len(cas):
        raise ValueError("Dos nombres de 'components' corresponden al mismo compuesto (mismo CAS).")
    m = _build(mdl, names, cas, zs, kij, use_library_kij, activity_parameters, warnings, assumptions)
    dropped = [nm for nm in all_names if nm not in names]
    if dropped:
        assumptions.append(f"Componentes con fracción molar 0 excluidos del cálculo: {', '.join(dropped)}.")
    notes: list[str] = []
    method = (f"Flash {kind} con thermo {thermo.__version__} (thermo.flash.FlashVLN: gas + dos líquidos candidatos, con "
              f"análisis de estabilidad). Modelo: {m['model_description']}.")
    base = {"specification": {"type": kind, **{k: given[k] for k in given}},
            "model": mdl, "components": names, "feed_mole_fractions": {nm: z for nm, z in zip(names, zs)},
            "pure_component_data": m["pure"], "binary_parameters": m["binary_parameters"]}
    not_included = [
        "Fases sólidas, hidratos, electrolitos y reacciones (ni la asociación en fase vapor de ácidos carboxílicos o HF).",
        "Más de dos fases líquidas.",
        "Propiedades de transporte de las fases.",
        "Ajuste de parámetros binarios: los kij o parámetros de actividad son los de la librería o los del usuario.",
    ]
    try:
        if kind == "TP":
            res = _tp_state(m, given["T"], given["P"], zs)
        elif kind in ("PH", "PS"):
            res = _solve_T(m, given["P"], zs, kind[1], given[kind[1]], notes)
        else:
            var, fixed = ("T", given["P"]) if kind == "PVF" else ("P", given["T"])
            try:
                spec = {"P": fixed} if var == "T" else {"T": fixed}
                res = _state_from_thermo(m["flasher"].flash(**spec, VF=given["VF"], zs=zs))
                guess = res.T if var == "T" else res.P
                ok = _state_is_valid(m, res, zs)
            except Exception as e:  # noqa: BLE001 — thermo's VF solver failed: same model, own bracketed solver below
                res, guess, ok = None, None, False
                notes.append(f"El solver de fracción de vapor de thermo falló ({type(e).__name__}).")
            if ok:
                notes.append(f"Flash {kind[:-2]}-VF: solver de thermo de equilibrio líquido-vapor; el estado pasa el "
                             "balance "
                             "de materia y la prueba de estabilidad independiente.")
            else:
                res = _solve_vf(m, zs, var, fixed, given["VF"], guess, notes)
    except Exception as e:  # noqa: BLE001 — non-convergence is reported, never hidden
        base.update(status="not_converged", phases=None, warnings=warnings + ["El flash no convergió; no hay resultado."],
                    assumptions=assumptions, not_included=not_included, method=method,
                    evidence={"verified": False, "reasons": [f"thermo no pudo resolver el flash {kind}: {type(e).__name__}: "
                                                             f"{str(e)[:300]}"]})
        return base
    T, P = float(res.T), float(res.P)
    table, objs = _phase_table(res, names, list(m["constants"].MWs))
    if res.solver == "own_three_phase_jump_resolution":
        warnings.append("Estado trifásico reconstruido en un salto isotérmico/isobárico (ver method): T o P quedan "
                        "determinadas con la resolución de la búsqueda, no exactamente; ver evidence para el equilibrio de "
                        "las tres fases.")
    elif res.solver != "thermo.FlashVLN":
        notes.append("El estado que devolvió thermo.FlashVLN no superaba la prueba de estabilidad independiente (le faltaba "
                     "una fase): se resolvió de nuevo con una sustitución sucesiva multifásica propia (Michelsen: "
                     "fracciones de fase por minimización de Q) sembrada con la fase de prueba inestable, con el mismo "
                     "modelo.")
        warnings.append("thermo.FlashVLN dio un estado inestable y el resultado procede del solver multifásico propio de "
                        "respaldo (mismo modelo termodinámico); ver evidence.")
    labels = [row["label"] for row in table]
    kvals = {}
    for a in range(len(table)):
        for b in range(a + 1, len(table)):
            xa, xb = objs[a][1], objs[b][1]
            kvals[f"{labels[a]}/{labels[b]}"] = {nm: (xa[i] / xb[i] if xb[i] > 0 else None) for i, nm in enumerate(names)}
    ev = _evidence(m, res, table, objs, zs, kind, given)
    vf = sum(row["mole_fraction_of_feed"] for row in table if row["type"] == "vapor")
    n_liq = sum(1 for row in table if row["type"] == "liquid")
    if not m["eos"]:
        for row, vp in zip(m["pure"], m["corr"].VaporPressures):
            lim = row.get("Psat_T_range_K")
            if lim and not lim[0] <= T <= lim[1]:
                warnings.append(f"{row['component']}: T = {T:.2f} K está fuera del intervalo de su presión de vapor "
                                f"({vp.method}, {lim[0]:.1f}-{lim[1]:.1f} K): valor EXTRAPOLADO; si el componente es "
                                "supercrítico el enfoque γ-φ no es fiable, usa una ecuación de estado.")
        if P > 1.0e6:
            warnings.append(f"P = {P / 1e5:.1f} bar con fase vapor ideal y sin Poynting: fuera del alcance de un modelo γ-φ "
                            "simple.")
    if n_liq >= 2:
        warnings.append("Se encontraron dos fases líquidas: las solubilidades mutuas son muy sensibles al modelo y a sus "
                        "parámetros (un kij o parámetros ajustados a ELV no suelen reproducir ELL).")
    if ev["stability"]["single_phase_feed"].get("stable") is False and len(table) == 1:
        warnings.append("El resultado tiene una sola fase pero la prueba de estabilidad independiente indica que se separa.")
    if not ev["verified"]:
        warnings.append("Resultado NO verificado: " + " ".join(ev["reasons"]))
    assumptions.append("Mezcla en equilibrio termodinámico; sin arrastre, cinética ni efectos de superficie.")
    assumptions.append("Constantes de componente puro (Tc, Pc, ω, Cp, Psat) de las bases de chemicals/thermo, con su método "
                       "en pure_component_data; no son datos del usuario.")
    assumptions.append("Etiquetas: V = vapor; L = único líquido; L1 = líquido menos denso, L2 = líquido más denso.")
    base.update(
        status="ok", solver=res.solver, T_K=T, P_Pa=P, vapor_fraction=vf, phase_count=len(table), phase_labels=labels,
        liquid_liquid_split=n_liq >= 2, phases=table, K_values=kvals,
        K_values_definition="K_i = fracción molar de i en la primera fase / fracción molar de i en la segunda (p. ej. V/L = "
                            "y_i/x_i)",
        H_J_mol=float(res.H()), S_J_molK=float(res.S()), enthalpy_reference=ENTHALPY_REFERENCE,
        warnings=warnings, assumptions=assumptions, not_included=not_included,
        method=method + (" " + " ".join(notes) if notes else ""), evidence=ev)
    return base


def _match_phases(ref: list[dict], ind: list[dict], names: list[str]) -> list[tuple[dict, dict | None]]:
    pairs, free = [], list(ind)
    for r in ref:
        cands = [p for p in free if p["type"] == r["type"]]
        if not cands:
            pairs.append((r, None))
            continue
        best = min(cands, key=lambda p: max(abs(p["mole_fractions"][nm] - r["x"][nm]) for nm in names))
        free.remove(best)
        pairs.append((r, best))
    return pairs + [(None, p) for p in free]


def compare_flash(reference: dict, components: list[str], mole_fractions: list[float], model: str | None = None,
                  T_K: float | None = None, P_Pa: float | None = None, vapor_fraction: float | None = None,
                  H_J_mol: float | None = None, S_J_molK: float | None = None, kij: list[dict] | None = None,
                  activity_parameters: list[dict] | None = None, use_library_kij: bool = True,
                  phase_fraction_tolerance: float = 0.01, mole_fraction_tolerance: float = 0.01,
                  temperature_tolerance_K: float = 0.5, pressure_relative_tolerance: float = 0.01) -> dict:
    """Compare another engine's flash result with `independent_flash` at the same specification. Engine-agnostic:
    `reference` = {"engine": str, "phases": [{"type": "vapor"|"liquid", "fraction": molar fraction of the feed,
    "mole_fractions": [...] in the order of `components` or {name: x}}], optional "T_K" / "P_Pa" of the result}."""
    if not isinstance(reference, dict) or not reference.get("phases"):
        raise ValueError("Falta 'reference' con 'phases': el resultado del otro motor que se quiere contrastar, como lista "
                         "de {'type': 'vapor'|'liquid', 'fraction': fracción molar de la alimentación, 'mole_fractions': "
                         "[...]}.")
    for tol_name, tol in (("phase_fraction_tolerance", phase_fraction_tolerance),
                          ("mole_fraction_tolerance", mole_fraction_tolerance),
                          ("temperature_tolerance_K", temperature_tolerance_K),
                          ("pressure_relative_tolerance", pressure_relative_tolerance)):
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(f"{tol_name} debe ser positivo.")
    ind = independent_flash(components, mole_fractions, model, T_K, P_Pa, vapor_fraction, H_J_mol, S_J_molK, kij,
                            activity_parameters, use_library_kij)
    names = ind["components"]
    all_names = [str(c) for c in components]
    ref_phases = []
    for k, ph in enumerate(reference["phases"]):
        if not isinstance(ph, dict) or ph.get("type") not in ("vapor", "liquid") or "fraction" not in ph \
                or "mole_fractions" not in ph:
            raise ValueError(f"reference['phases'][{k}] necesita 'type' ('vapor' o 'liquid'), 'fraction' y "
                             "'mole_fractions'.")
        mf = ph["mole_fractions"]
        if isinstance(mf, dict):
            low = {str(a).strip().lower(): float(v) for a, v in mf.items()}
            missing = [nm for nm in names if nm.strip().lower() not in low]
            if missing:
                raise ValueError(f"reference['phases'][{k}]['mole_fractions'] no tiene: {', '.join(missing)}.")
            x = {nm: low[nm.strip().lower()] for nm in names}
        else:
            if len(mf) != len(all_names):
                raise ValueError(f"reference['phases'][{k}]['mole_fractions'] debe tener un valor por componente, en el "
                                 "orden de 'components'.")
            x = {nm: float(v) for nm, v in zip(all_names, mf) if nm in names}
        if abs(sum(x.values()) - 1.0) > 1e-3:
            raise ValueError(f"reference['phases'][{k}]: las fracciones molares suman {sum(x.values()):.5g}, no 1.")
        if float(ph["fraction"]) > 0.0:
            ref_phases.append({"type": ph["type"], "fraction": float(ph["fraction"]), "x": x})
    if abs(sum(p["fraction"] for p in ref_phases) - 1.0) > 1e-3:
        raise ValueError("reference: las fracciones de fase no suman 1 (deben ser fracciones molares de la alimentación).")
    out = {"reference_engine": reference.get("engine"), "independent": ind,
           "tolerances": {"phase_fraction_abs": phase_fraction_tolerance, "mole_fraction_abs": mole_fraction_tolerance,
                          "temperature_K": temperature_tolerance_K, "pressure_rel": pressure_relative_tolerance}}
    warnings: list[str] = []
    assumptions = [
        "Las tolerancias de la comparación son criterios numéricos de esta herramienta (modificables), no incertidumbres de "
        "los datos.",
        "Coincidir con el mismo modelo en los dos motores certifica el cálculo, no que el modelo represente la mezcla; "
        "discrepar puede deberse a constantes de componente puro o parámetros binarios distintos entre motores (ver "
        "pure_component_data y binary_parameters del flash independiente).",
    ]
    not_included = ["Comparación de entalpías y entropías: cada motor usa su propio estado de referencia.",
                    "Comparación de densidades y propiedades de transporte."]
    if ind["status"] != "ok":
        out.update(verdict="independent_not_converged", differences=None, warnings=ind["warnings"], assumptions=assumptions,
                   not_included=not_included, method="Comparación no realizada: el flash independiente no convergió.",
                   evidence={"verified": False, "reasons": ind["evidence"]["reasons"]})
        return out
    ind_phases = [p for p in ind["phases"] if p["mole_fraction_of_feed"] > 1e-12]
    pairs = _match_phases(ref_phases, ind_phases, names)
    diffs, max_dbeta, max_dx = [], 0.0, 0.0
    same_set = all(r is not None and p is not None for r, p in pairs)
    for r, p in pairs:
        if r is None or p is None:
            diffs.append({"type": (r or p)["type"], "only_in": "reference" if p is None else "independent",
                          "fraction": r["fraction"] if p is None else p["mole_fraction_of_feed"],
                          "label_independent": p["label"] if p else None})
            continue
        dx = {nm: p["mole_fractions"][nm] - r["x"][nm] for nm in names}
        dbeta = p["mole_fraction_of_feed"] - r["fraction"]
        max_dbeta, max_dx = max(max_dbeta, abs(dbeta)), max(max_dx, max(abs(v) for v in dx.values()))
        diffs.append({"type": r["type"], "label_independent": p["label"], "fraction_reference": r["fraction"],
                      "fraction_independent": p["mole_fraction_of_feed"], "fraction_difference": dbeta,
                      "mole_fraction_difference": dx, "max_abs_mole_fraction_difference": max(abs(v) for v in dx.values())})
    state = {}
    ok_state = True
    if reference.get("T_K") is not None:
        state["T_difference_K"] = ind["T_K"] - float(reference["T_K"])
        ok_state &= abs(state["T_difference_K"]) <= temperature_tolerance_K
    if reference.get("P_Pa") is not None:
        state["P_relative_difference"] = ind["P_Pa"] / float(reference["P_Pa"]) - 1.0
        ok_state &= abs(state["P_relative_difference"]) <= pressure_relative_tolerance
    if not ind["evidence"]["verified"]:
        verdict = "independent_not_verified"
        warnings.append("El flash independiente no está verificado: la comparación no es concluyente.")
    elif not same_set:
        verdict = "phase_set_differs"
        warnings.append("Los dos motores no encuentran el mismo conjunto de fases; revisa en el otro motor si el flash hace "
                        "análisis de estabilidad (segundo líquido) y si usa el mismo modelo y parámetros.")
    elif max_dbeta <= phase_fraction_tolerance and max_dx <= mole_fraction_tolerance and ok_state:
        verdict = "agree"
    else:
        verdict = "disagree"
        warnings.append("Mismas fases pero diferencias por encima de la tolerancia: compara constantes críticas, ω y "
                        "parámetros binarios de ambos motores antes de concluir cuál está mal.")
    out.update(verdict=verdict, phase_set_matches=same_set,
               max_abs_phase_fraction_difference=max_dbeta if same_set else None,
               max_abs_mole_fraction_difference=max_dx if same_set else None, state_differences=state, differences=diffs,
               warnings=warnings + [w for w in ind["warnings"]], assumptions=assumptions, not_included=not_included,
               method="Flash independiente con thermo en la misma especificación y emparejamiento de fases por tipo y "
                      "composición más cercana; diferencias = independiente − referencia.",
               evidence={"verified": ind["evidence"]["verified"], "reasons": ind["evidence"]["reasons"],
                         "independent_flash": {k: ind["evidence"][k] for k in ("material_balance_max_abs",
                                                                               "fugacity_equality_max_abs_dlnf",
                                                                               "rachford_rice_max_abs_residual")},
                         "meaning": "verified se refiere al flash independiente usado como patrón, no al motor de "
                                    "referencia"})
    return out
