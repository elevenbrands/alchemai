"""Equation-oriented distillation optimisation (bloque M8a of docs/MAPA_MOTORES.md): IDAES TrayColumn solved
simultaneously by ipopt (idaes-ext 3.4.2 binaries, installed with `idaes get-extensions`, EPL-2.0 + HSL for IDAES).

DWSIM solves a column sequentially: an optimiser around it needs one full column solve per trial (optimize_design_multi).
Here the whole column (MESH equations on every tray, condenser, reboiler) and the purity constraints are ONE nonlinear
program, so ipopt returns the minimum reboiler duty, the optimal reflux and boil-up ratios AND the Lagrange multipliers:
how many MW each extra unit of purity costs.

Systems (M8b): any mixture the user names whose VLE is ideal (Raoult + ideal gas). The IDAES property package is
GENERATED from `chemicals` / `thermo` tables (packages/eo/ideal_properties.py, sources cited in the output), screened
with Dortmund UNIFAC (azeotrope or |ln γi/γj| > 0.10 → refused) and checked against DIPPR on every call. The M8a
package (IDAES `BT_ideal` with the liquid cp corrected) stays available as property_package="idaes_BT_ideal".

IDAES is BSD and runs in this process; ipopt runs as a separate executable launched by Pyomo, with max_cpu_time.

Evidence → seal (`verified` only when all hold):
  * ipopt_optimal: termination condition optimal
  * specs_met: each purity ≥ its spec − SPEC_TOL
  * local_optimum: square solves at the 8 neighbours (reflux, boil-up) × (1 ± PERTURB): every neighbour that meets
    the specs has a duty ≥ the optimum (− 1e-9 relative)
  * duals_consistent: for each active spec, re-optimising with the spec tightened by DELTA_SPEC moves the duty by the
    multiplier × DELTA_SPEC within DUAL_REL_TOL
  * properties_verified (generated package): property_check within tolerance over the column's temperature range
  * inside_property_window: product temperatures strictly inside the package's temperature bounds
The reference case of the IDAES test suite (TestBTXIdealGeneric) is reproduced in tests/test_eo_column.py.
"""
from __future__ import annotations

import time

SPEC_TOL = 1e-6
PERTURB = 0.01
DELTA_SPEC = 1e-4
DUAL_REL_TOL = 0.05
MAX_CPU_S = 120
MAX_ITER = 3000
COMPONENTS = ("benzene", "toluene")          # the M8a package
PACKAGES = ("chemicals", "idaes_BT_ideal")
STREAMS = ("distillate", "bottoms")
KINDS = ("mole_fraction", "recovery")
ACTIVITY_MODELS = ("ideal", "NRTL")
BUBBLE_T_TOL_K = 0.05        # IDAES product T vs thermo bubble T with the same NRTL parameters and Psat
SMOOTH_EPS_STEPS = (1e-3, 1e-4)  # K; SmoothVLE eps_1 continued from IDAES's 0.01 on the NRTL path (eps_2 = eps_1/20)
SMOOTH_EPS_EXTRA = (3e-5, 1e-5, 3e-6, 1e-6)   # K; further steps taken only while the balance/phantom seal fails (M15b)
SPEC_CONTINUATION_MIN_STEP = 1e-4             # smallest fraction of the spec path tried before giving up (M15b)
PHANTOM_VF_TOL = 1e-3        # max vapour fraction SmoothVLE leaves in a saturated-liquid inlet (reflux, tray liquid)
BALANCE_TOL = 1e-4           # relative closure of each component, feed vs distillate + bottoms


def ipopt_available() -> bool:
    try:
        from pyomo.environ import SolverFactory
        import idaes  # noqa: F401  (puts ~/.idaes/bin on the solver path)
        return bool(SolverFactory("ipopt").available(exception_flag=False))
    except Exception:  # noqa: BLE001
        return False


def _solver():
    from idaes.core.solvers import get_solver
    if not ipopt_available():
        raise RuntimeError("ipopt no está instalado: corre `uv run idaes get-extensions` (binarios oficiales de IDAES en "
                           "~/.idaes/bin; sin costo).")
    return get_solver("ipopt", options={"max_cpu_time": MAX_CPU_S, "max_iter": MAX_ITER})


def bt_ideal_configuration(corrected: bool = True) -> dict:
    """IDAES `BT_ideal` configuration. corrected=True fixes its liquid heat capacity: the Perry's coefficients
    (benzene 1.29e2, toluene 1.40e2 …, i.e. 129 J/mol·K) are declared in J/kmol·K, 1000× too small, so the liquid
    enthalpy is flat and ΔHvap grows with T (benzene 38.8 kJ/mol at 354 K vs 30.7 DIPPR). Same numbers, J/mol units."""
    import copy
    from pyomo.environ import units as pyunits
    from idaes.models.properties.modular_properties.examples.BT_ideal import configuration
    cfg = copy.deepcopy(configuration)
    if corrected:
        for comp in cfg["components"].values():
            coeffs = comp["parameter_data"]["cp_mol_liq_comp_coeff"]
            for i, k in enumerate(("1", "2", "3", "4", "5")):
                val = coeffs[k][0]
                coeffs[k] = (val, pyunits.J / pyunits.mol / pyunits.K ** (i + 1))
    return cfg


def _set_smooth_eps(m, eps: tuple[float, float]) -> int:
    """SmoothVLE smoothing temperatures eps_1 (Teq = smooth_max(T, Tbub)) and eps_2 (smooth_min(·, Tdew)) of every state
    block under m. IDAES defaults 0.01 / 0.0005 K. A saturated liquid (condenser outlet, reflux, liquid passed between
    trays) is evaluated at Teq ≈ Tbub + eps_1/2, so a phantom vapour fraction ≈ (eps_1/2)/(Tdew − Tbub) appears: the
    total condenser then drops it from the component balance (methanol-water 0.18 %, ethanol-water near the azeotrope
    1.1 %) and every liquid inlet carries vapour enthalpy. Harmless for wide-boiling mixtures, not near a pinch (M15)."""
    from pyomo.environ import Param
    n = 0
    for prm in m.component_objects(Param, descend_into=True):
        if prm.local_name.startswith("eps_1_"):
            prm.set_value(eps[0])
            n += 1
        elif prm.local_name.startswith("eps_2_"):
            prm.set_value(eps[1])
            n += 1
    return n


def phantom_vapour(m) -> float:
    """Largest vapour fraction in the liquid inlets of the trays and the condenser outlet (should be ~0)."""
    from pyomo.environ import value
    u = m.fs.unit
    blocks = [u.condenser.control_volume.properties_out[0]]
    for sec in (u.rectification_section, u.stripping_section):
        blocks += [sec[i].properties_in_liq[0] for i in sec]
    blocks.append(u.feed_tray.properties_in_liq[0])
    return max(value(b.phase_frac["Vap"]) for b in blocks)


def _sealed(m) -> bool:
    """Component balance and phantom vapour inside their tolerances (the two checks SmoothVLE's eps can break)."""
    return (all(abs(v) <= BALANCE_TOL for v in component_balance(m).values())
            and phantom_vapour(m) <= PHANTOM_VF_TOL)


def _optimise_tightening(m, specs, bounds, solver):
    """_optimise at IDAES's SmoothVLE eps (robust), then re-optimise warm while eps_1 goes down SMOOTH_EPS_STEPS (a
    small eps from a cold start does not converge). A failed step restores the last optimum and its eps; the evidence
    then reports the phantom vapour left (phantom_vapour) against PHANTOM_VF_TOL."""
    ok, term, duals = _optimise(m, specs, bounds, solver)
    reached = (0.01, 0.0005)
    if not ok:
        return ok, term, duals, list(reached)
    snap = _snapshot(m)
    for e1 in SMOOTH_EPS_STEPS:
        e = (e1, e1 / 20)
        _set_smooth_eps(m, e)
        ok2, term2, duals2 = _optimise(m, specs, bounds, solver)
        if ok2:
            reached, term, duals, snap = e, term2, duals2, _snapshot(m)
        else:
            _restore(snap)
            _set_smooth_eps(m, reached)
            break
    return ok, term, duals, list(reached)


def _tighten_square(m, solver) -> list[float]:
    """Same eps continuation for a square (simulation) model."""
    reached, snap = (0.01, 0.0005), _snapshot(m)
    for e1 in SMOOTH_EPS_STEPS:
        e = (e1, e1 / 20)
        _set_smooth_eps(m, e)
        if _square(m, solver):
            reached, snap = e, _snapshot(m)
        else:
            _restore(snap)
            _set_smooth_eps(m, reached)
            break
    return list(reached)


def component_balance(m) -> dict:
    """Overall feed − (distillate + bottoms) per component, relative to the feed (closes to the solver tolerance when
    the model is consistent)."""
    from pyomo.environ import value
    u = m.fs.unit
    d, b = u.condenser.distillate, u.reboiler.bottoms
    out = {}
    for c in m.fs.properties.component_list:
        fin = value(u.feed.flow_mol[0] * u.feed.mole_frac_comp[0, c])
        prod = value(d.flow_mol[0] * d.mole_frac_comp[0, c] + b.flow_mol[0] * b.mole_frac_comp[0, c])
        out[c] = (prod - fin) / fin
    return out


def build_column(n_trays: int, feed_tray: int, feed: dict, pressure_Pa: float, corrected: bool = True, config: dict | None = None,
                 smooth_eps: tuple[float, float] | None = None):
    """IDAES TrayColumn (total condenser at bubble point, partial reboiler). feed = {flow_mol_s, T_K, P_Pa, z: {comp: x}}
    (or the M8a form with x_benzene). config=None → the M8a benzene-toluene package (corrected or stock)."""
    from pyomo.environ import ConcreteModel
    from idaes.core import FlowsheetBlock
    from idaes.models_extra.column_models import TrayColumn
    from idaes.models_extra.column_models.condenser import CondenserType, TemperatureSpec
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    z = feed.get("z") or {"benzene": feed["x_benzene"], "toluene": 1.0 - feed["x_benzene"]}
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = GenericParameterBlock(**(config if config is not None else bt_ideal_configuration(corrected)))
    m.fs.unit = TrayColumn(number_of_trays=n_trays, feed_tray_location=feed_tray, condenser_type=CondenserType.totalCondenser,
                           condenser_temperature_spec=TemperatureSpec.atBubblePoint, property_package=m.fs.properties,
                           has_heat_transfer=False, has_pressure_change=False)
    u = m.fs.unit
    u.feed.flow_mol.fix(feed["flow_mol_s"])
    u.feed.temperature.fix(feed["T_K"])
    u.feed.pressure.fix(feed["P_Pa"])
    for c, x in z.items():
        u.feed.mole_frac_comp[0, c].fix(x)
    u.condenser.condenser_pressure.fix(pressure_Pa)
    if smooth_eps is not None:
        _set_smooth_eps(m, smooth_eps)
    return m


def _state(m) -> dict:
    from pyomo.environ import value
    u = m.fs.unit
    d, b = u.condenser.distillate, u.reboiler.bottoms
    comps = list(m.fs.properties.component_list)
    out = {"reflux_ratio": value(u.condenser.reflux_ratio), "boilup_ratio": value(u.reboiler.boilup_ratio),
           "reboiler_duty_W": value(u.reboiler.heat_duty[0]), "condenser_duty_W": value(u.condenser.heat_duty[0])}
    for key, st in (("distillate", d), ("bottoms", b)):
        out[key] = {"flow_mol_s": value(st.flow_mol[0]), "T_K": value(st.temperature[0]),
                    "mole_fractions": {c: value(st.mole_frac_comp[0, c]) for c in comps}}
    if set(comps) == set(COMPONENTS):          # M8a keys
        out["distillate"]["x_benzene"] = out["distillate"]["mole_fractions"]["benzene"]
        out["bottoms"]["x_toluene"] = out["bottoms"]["mole_fractions"]["toluene"]
    return out


def simulate_column(n_trays: int, feed_tray: int, feed: dict, pressure_Pa: float, reflux_ratio: float,
                    boilup_ratio: float, corrected: bool = True, config: dict | None = None) -> dict:
    """Square solve (no optimisation). corrected=False is IDAES's own configuration (its test-suite reference)."""
    from pyomo.environ import check_optimal_termination
    import idaes.logger as idaeslog
    m = build_column(n_trays, feed_tray, feed, pressure_Pa, corrected, config)
    m.fs.unit.condenser.reflux_ratio.fix(reflux_ratio)
    m.fs.unit.reboiler.boilup_ratio.fix(boilup_ratio)
    m.fs.unit.initialize(outlvl=idaeslog.CRITICAL)
    res = _solver().solve(m)
    if not check_optimal_termination(res):
        raise RuntimeError(f"ipopt no resolvió la columna: {res.solver.termination_condition}")
    return _state(m)


def simulate_column_nrtl(components: list[str], feed: dict, n_trays: int, feed_tray: int, pressure_Pa: float,
                         reflux_ratio: float, distillate_mass_flow_kg_s: float | None = None,
                         distillate_flow_mol_s: float | None = None, binary_parameters=None,
                         excess_enthalpy: bool = True) -> dict:
    """Square NRTL column at a given reflux ratio and distillate rate (molar or mass, as a published case states it):
    the M15 reference-case solve. feed = {flow_mol_s, T_K, P_Pa, z}."""
    from pyomo.environ import value
    from packages.eo import nrtl_properties as NP
    if (distillate_mass_flow_kg_s is None) == (distillate_flow_mol_s is None):
        raise ValueError("da distillate_mass_flow_kg_s o distillate_flow_mol_s (uno)")
    params = NP.resolve_parameters(components, binary_parameters)
    cfg, prov = NP.build_configuration(components, pressure_Pa, params, extra_T=[feed["T_K"]],
                                       excess_enthalpy=excess_enthalpy)
    solver = _solver()
    bounds = {"reflux_ratio": (0.1, 50.0), "boilup_ratio": (0.01, 50.0)}
    mw = {c: prov["components"][c]["MW_g_mol"] * 1e-3 for c in components}
    if distillate_flow_mol_s is None:       # first guess of D in mol/s from the mass rate at the feed's light end
        light = min(components, key=lambda c: prov["components"][c]["Tc_K"])
        D0 = distillate_mass_flow_kg_s / mw[light]
    else:
        D0 = distillate_flow_mol_s
    B0 = feed["flow_mol_s"] - D0
    br0 = (reflux_ratio + 1.0) * D0 / B0
    m, start = _initial_point(n_trays, feed_tray, feed, pressure_Pa, cfg, bounds, solver, params=params,
                              extra_starts=[(reflux_ratio, br0)])
    u = m.fs.unit
    u.condenser.reflux_ratio.fix(reflux_ratio)
    d = u.condenser.distillate
    molar = distillate_flow_mol_s is not None
    target = distillate_flow_mol_s if molar else distillate_mass_flow_kg_s
    _hold_distillate(m, value(d.flow_mol[0]))
    ok = _square(m, solver)
    eps = _tighten_square(m, solver)
    # continue the held distillate (mol/s, or the mass rate through the composition) to the target
    for _ in range(60):
        cur = value(d.flow_mol[0]) if molar else value(sum(d.flow_mol[0] * d.mole_frac_comp[0, c] * mw[c]
                                                           for c in components))
        if abs(cur / target - 1) < 1e-10:
            break
        snap = _snapshot(m)
        Dnow = value(d.flow_mol[0])
        step = 1.0
        while step > 1e-3:
            Dtry = Dnow * (1 + step * (target / cur - 1))
            m.hold_distillate.set_value(d.flow_mol[0] == Dtry)
            if _square(m, solver):
                break
            _restore(snap)
            step *= 0.5
        else:
            raise RuntimeError("la columna no llega al destilado pedido")
    if not ok and not _square(m, solver):
        raise RuntimeError("ipopt no resolvió la columna NRTL")
    st = _state(m)
    st["smooth_eps_K"] = eps
    st["phantom_vapour_fraction"] = phantom_vapour(m)
    st["distillate"]["mass_flow_kg_s"] = sum(st["distillate"]["flow_mol_s"] * st["distillate"]["mole_fractions"][c] *
                                             mw[c] for c in components)
    st["bottoms"]["mass_flow_kg_s"] = sum(st["bottoms"]["flow_mol_s"] * st["bottoms"]["mole_fractions"][c] * mw[c]
                                          for c in components)
    st["initial_point"] = start
    st["component_balance_rel"] = component_balance(m)
    st["binary_parameters"] = params["pairs"]
    st["thermo_bubble_check"] = NP.thermo_bubble_points(params, pressure_Pa, {k: st[k]["mole_fractions"]
                                                                              for k in STREAMS})
    return st


def _spec_key(sp: dict) -> str:
    return f"{sp['kind']}:{sp['component']}@{sp['stream']}"


def _spec_expr(m, sp: dict):
    u = m.fs.unit
    st = u.condenser.distillate if sp["stream"] == "distillate" else u.reboiler.bottoms
    x = st.mole_frac_comp[0, sp["component"]]
    if sp["kind"] == "mole_fraction":
        return x
    return st.flow_mol[0] * x / (u.feed.flow_mol[0] * u.feed.mole_frac_comp[0, sp["component"]])


def _spec_value(state: dict, feed: dict, sp: dict) -> float:
    st = state[sp["stream"]]
    x = st["mole_fractions"][sp["component"]]
    return x if sp["kind"] == "mole_fraction" else st["flow_mol_s"] * x / (feed["flow_mol_s"] * feed["z"][sp["component"]])


def _optimise(m, specs: list[dict], bounds: dict, solver):
    from pyomo.environ import ConstraintList, Objective, Suffix, minimize, check_optimal_termination
    u = m.fs.unit
    for v, (lo, hi) in ((u.condenser.reflux_ratio, bounds["reflux_ratio"]), (u.reboiler.boilup_ratio, bounds["boilup_ratio"])):
        v.unfix()
        v.setlb(lo)
        v.setub(hi)
    for name in ("specs", "obj", "dual"):
        if hasattr(m, name):
            m.del_component(name)
    m.specs = ConstraintList()
    cons = [m.specs.add(_spec_expr(m, sp) >= sp["min"]) for sp in specs]
    m.obj = Objective(expr=u.reboiler.heat_duty[0] * 1e-6, sense=minimize)
    m.dual = Suffix(direction=Suffix.IMPORT)
    try:
        res = solver.solve(m)
    except ValueError as ex:          # ipopt evaluation error: Pyomo cannot load a result with status "error"
        return False, f"error de ipopt ({ex})", {_spec_key(sp): float("nan") for sp in specs}
    ok = check_optimal_termination(res)
    # ipopt's multiplier for g(x) ≥ b is ≥ 0 in Pyomo's sign convention for a minimisation: d(obj)/d(b)
    duals = {_spec_key(sp): abs(m.dual.get(c, 0.0)) for sp, c in zip(specs, cons)}
    return ok, str(res.solver.termination_condition), duals


def _validate(components, feed_flow_mol_s, feed_T_K, feed_P_Pa, feed_mole_fractions, n_trays, feed_tray, pressure_Pa,
              specs, property_package):
    if property_package not in PACKAGES:
        raise ValueError(f"property_package must be one of {PACKAGES}")
    if not isinstance(components, (list, tuple)) or len(components) < 2 or len(set(components)) != len(components):
        raise ValueError("components: lista de 2 o más compuestos distintos")
    if property_package == "idaes_BT_ideal" and set(components) != set(COMPONENTS):
        raise ValueError("idaes_BT_ideal solo tiene benceno y tolueno (components=['benzene', 'toluene'])")
    for k, v in dict(feed_flow_mol_s=feed_flow_mol_s, feed_T_K=feed_T_K, feed_P_Pa=feed_P_Pa, pressure_Pa=pressure_Pa).items():
        if v is None or not v > 0:
            raise ValueError(f"{k} debe ser positivo (sin valor por defecto)")
    if not isinstance(feed_mole_fractions, dict) or set(feed_mole_fractions) != set(components):
        raise ValueError("feed_mole_fractions: una fracción por cada compuesto de components")
    if any(not 0 < x < 1 for x in feed_mole_fractions.values()) or abs(sum(feed_mole_fractions.values()) - 1) > 1e-6:
        raise ValueError("feed_mole_fractions: cada una en (0, 1) y que sumen 1")
    if n_trays < 2 or not (1 <= feed_tray <= n_trays):
        raise ValueError("feed_tray debe estar entre 1 (plato superior) y n_trays")
    if not specs:
        raise ValueError("specs: al menos una especificación de pureza o recuperación")
    for sp in specs:
        if sp.get("stream") not in STREAMS or sp.get("kind") not in KINDS or sp.get("component") not in components:
            raise ValueError(f"spec inválida {sp}: stream en {STREAMS}, kind en {KINDS}, component en components")
        if not 0 < sp.get("min", -1) < 1:
            raise ValueError(f"spec {sp}: min en (0, 1)")
    if len({_spec_key(sp) for sp in specs}) != len(specs):
        raise ValueError("specs repetidas")


def eo_column_optimize(components: list[str], feed_flow_mol_s: float, feed_T_K: float, feed_P_Pa: float,
                       feed_mole_fractions: dict, n_trays: int, feed_tray: int, pressure_Pa: float, specs: list[dict],
                       reflux_ratio_bounds: tuple[float, float] = (0.1, 20.0),
                       boilup_ratio_bounds: tuple[float, float] = (0.1, 20.0), evidence: bool = True,
                       property_package: str = "chemicals", activity_model: str = "ideal",
                       binary_parameters: list | dict | None = None, excess_enthalpy: bool = True) -> dict:
    """Minimum reboiler duty of a tray column meeting purity/recovery specs, solved as one NLP (IDAES + ipopt), with a
    property package generated from chemicals/thermo. spec = {"stream": "distillate"|"bottoms", "component": name,
    "kind": "mole_fraction"|"recovery", "min": value}.
    activity_model="ideal" (default, M8b): Raoult, refused if UNIFAC finds the mixture non-ideal.
    activity_model="NRTL" (M15): γ from NRTL with binary parameters from ONE labelled source per pair — the user's
    (binary_parameters=[{"components": [i, j], "b_ij_K", "b_ji_K" | "dg_ij_J_mol", "dg_ji_J_mol" | "dg_ij_cal_mol",
    "dg_ji_cal_mol", optional "a_ij", "a_ji", "alpha", "source"}]), the result dict of vle_binary_regression (passed
    as is, alone or inside the list), or the ChemSep NRTL databank of thermo; refused if none has the pair.
    excess_enthalpy (NRTL only, M15b): True adds NRTL's h^E (Gibbs-Helmholtz, same parameters) to the liquid enthalpy;
    False keeps Σ x·h_i. Its weight in the duty is reported in properties.excess_enthalpy either way."""
    from packages.eo import ideal_properties as IP
    _validate(components, feed_flow_mol_s, feed_T_K, feed_P_Pa, feed_mole_fractions, n_trays, feed_tray, pressure_Pa,
              specs, property_package)
    if activity_model not in ACTIVITY_MODELS:
        raise ValueError(f"activity_model debe ser uno de {ACTIVITY_MODELS}")
    if activity_model != "ideal" and property_package != "chemicals":
        raise ValueError("activity_model NRTL solo con property_package='chemicals'")
    if activity_model == "ideal" and binary_parameters:
        raise ValueError("binary_parameters solo aplica con activity_model='NRTL'")
    components = list(components)
    t0 = time.perf_counter()
    t_sat = {c: IP.saturation_T(c, pressure_Pa) for c in components}
    feed = {"flow_mol_s": feed_flow_mol_s, "T_K": feed_T_K, "P_Pa": feed_P_Pa, "z": dict(feed_mole_fractions)}
    if activity_model == "NRTL":
        return _eo_nrtl(components, feed, n_trays, feed_tray, pressure_Pa, specs, reflux_ratio_bounds,
                        boilup_ratio_bounds, evidence, binary_parameters, t_sat, t0, excess_enthalpy)
    ideality = IP.ideality_check(components, pressure_Pa)
    if not ideality["ideal"]:
        raise ValueError("mezcla no ideal: con activity_model='ideal' solo se resuelve VLE ideal (Raoult + gas ideal). "
                         + "; ".join(ideality["reasons"]) + ". Usa activity_model='NRTL' con parámetros binarios (tuyos, "
                         "de vle_binary_regression o del banco ChemSep), o la columna de DWSIM (optimize_design_multi).")
    if property_package == "chemicals":
        cfg, prov = IP.build_configuration(components, pressure_Pa, extra_T=[feed_T_K])
        lo, hi = min(t_sat.values()), max(t_sat.values())
        pcheck = IP.property_check(cfg, components, [lo + f * (hi - lo) for f in (0.0, 1 / 3, 2 / 3, 1.0)])
    else:
        cfg, pcheck = bt_ideal_configuration(True), None
        prov = {"package": "IDAES BT_ideal (Perry's/RPP4) con cp líquido corregido (M8a)", "temperature_window_K": (273.15, 450.0)}
    bounds = {"reflux_ratio": tuple(reflux_ratio_bounds), "boilup_ratio": tuple(boilup_ratio_bounds)}
    solver = _solver()
    m, start = _initial_point(n_trays, feed_tray, feed, pressure_Pa, cfg, bounds, solver)
    ok, term, duals = _optimise(m, specs, bounds, solver)
    if not ok:
        raise RuntimeError(f"ipopt no encontró un óptimo ({term}): las especificaciones pueden ser inalcanzables con "
                           f"{n_trays} platos dentro de los límites de reflujo/boil-up {bounds}")
    opt = _state(m)
    out = {"components": components, "optimum": opt, "reboiler_duty_MW": opt["reboiler_duty_W"] / 1e6,
           "condenser_duty_MW": opt["condenser_duty_W"] / 1e6,
           "specs": [sp | {"value": _spec_value(opt, feed, sp)} for sp in specs],
           "duals_MW_per_unit_spec": duals,
           "active_specs": [k for k, v in duals.items() if v > 1e-6],
           "column": {"n_trays": n_trays, "feed_tray_from_top": feed_tray, "condenser": "total, at bubble point",
                      "reboiler": "partial", "pressure_Pa": pressure_Pa, "stages_incl_condenser_reboiler": n_trays + 2},
           "saturation_T_K_at_column_P": t_sat, "initial_point": start,
           "properties": {"package": property_package, "provenance": prov, "property_check": pcheck,
                          "ideality_check": ideality},
           "engine": "IDAES TrayColumn (modular generic properties) + ipopt (idaes-ext)",
           "assumptions": [
               "VLE ideal (Raoult + gas ideal): comprobado con UNIFAC Dortmund por pares (sin azeótropo, |ln γi/γj| ≤ "
               f"{IP.MAX_LN_GAMMA_RATIO}) y |B·P/RT| ≤ {IP.MAX_B_P_RT} (Tsonopoulos); UNIFAC es una estimación, no un dato "
               "experimental del sistema.",
               "Paquete de propiedades generado de chemicals/thermo (fuentes en properties.provenance.sources): Psat Wagner "
               "(McGarry), cp gas ideal (Poling), densidad DIPPR 105, cp líquido ajustado a thermo, ΔHvap = DIPPR 106 en Tsat "
               "de cada compuesto a la presión de la columna (la entalpía de referencia del líquido es efectiva, no la de "
               "formación)." if property_package == "chemicals" else
               "Paquete BT_ideal de IDAES (Perry's / RPP4) con el cp líquido corregido (IDAES lo declara en J/kmol·K: 1000× menor).",
               "Platos de equilibrio (eficiencia 100 %), sin caída de presión ni pérdidas de calor.",
               "Objetivo = carga del rehervidor; variables = reflujo y boil-up dentro de los límites dados (límites del "
               "solver, no datos del usuario)."]}
    if evidence:
        out["evidence"] = _evidence(m, cfg, opt, duals, specs, bounds, solver, n_trays, feed_tray, feed, pressure_Pa,
                                    pcheck, prov["temperature_window_K"])
    out["seconds"] = time.perf_counter() - t0
    return out


def _eo_nrtl(components, feed, n_trays, feed_tray, pressure_Pa, specs, rr_bounds, br_bounds, evidence,
             binary_parameters, t_sat, t0, excess_enthalpy=True) -> dict:
    """activity_model='NRTL' path (M15): screening → generated package with NRTL liquid → same NLP and evidence."""
    from packages.eo import ideal_properties as IP
    from packages.eo import nrtl_properties as NP
    try:
        params = NP.resolve_parameters(components, binary_parameters)
    except NP.MissingParameters as ex:
        raise ValueError(str(ex)) from ex
    virial, vreasons = IP.virial_check(components, pressure_Pa)
    if vreasons:
        raise ValueError("el vapor no se puede tratar como gas ideal: " + "; ".join(vreasons) + f" (tolerancia "
                         f"|B·P/RT| ≤ {IP.MAX_B_P_RT}). Usa DWSIM con una ecuación de estado para el vapor.")
    azeos, areasons = NP.azeotrope_refusals(params, pressure_Pa, feed["z"], specs)
    if areasons:
        raise ValueError("especificaciones inalcanzables por destilación simple: " + "; ".join(areasons) +
                         ". Cambia las especificaciones al lado del azeótropo de la alimentación, o usa destilación "
                         "azeotrópica/extractiva o por cambio de presión en DWSIM.")
    from packages.eo import shortcut as SC
    shortcut = SC.binary_shortcut(params, pressure_Pa, feed, specs, n_trays, feed_tray, tuple(rr_bounds),
                                  tuple(br_bounds), azeos)
    if shortcut and shortcut["refusals"]:
        raise ValueError("especificaciones inalcanzables con esta columna: " + "; ".join(shortcut["refusals"]))
    cross = NP.unifac_cross_check(params, pressure_Pa)
    cfg, prov = NP.build_configuration(components, pressure_Pa, params, extra_T=[feed["T_K"]],
                                       excess_enthalpy=excess_enthalpy)
    lo, hi = min(t_sat.values()), max(t_sat.values())
    grid_T = [lo + f * (hi - lo) for f in (0.0, 1 / 3, 2 / 3, 1.0)]
    pcheck = IP.property_check(cfg, components, grid_T)
    gcheck = NP.nrtl_check(cfg, params, grid_T)
    bounds = {"reflux_ratio": tuple(rr_bounds), "boilup_ratio": tuple(br_bounds)}
    solver = _solver()
    m, start = _initial_point(n_trays, feed_tray, feed, pressure_Pa, cfg, bounds, solver, params=params,
                              extra_starts=_spec_start_points(feed, specs),
                              homotopy_starts=shortcut["starts"] if shortcut else ())
    snap_start = _snapshot(m)
    ok, term, duals, eps = False, "sin intentar", {}, [0.01, 0.0005]
    if len(specs) == 2:
        two = _solve_two_specs(m, specs, feed, bounds, solver)
        start["spec_continuation"] = two["info"]
        ok, term, duals, eps = two["ok"], two["termination"], two["duals"], two["eps"]
    square_route = ok
    duals_source = start.get("spec_continuation", {}).get("duals_source") if ok else None
    fd_duals = bool(duals_source and duals_source.startswith("sensibilidades"))
    duals_source = duals_source or "multiplicadores de ipopt"
    if not ok:
        # M15 route (the NLP straight from the start): 1 or 3+ specs, or the square route gave up
        _restore(snap_start)
        _set_smooth_eps(m, (0.01, 0.0005))
        m.fs.unit.condenser.reflux_ratio.fix(start["reflux_ratio"])
        m.fs.unit.reboiler.boilup_ratio.fix(start["boilup_ratio"])
        if len(specs) == 2:
            start["spec_continuation"]["fell_back_to_nlp_because"] = term
        ok, term, duals, eps = _optimise_tightening(m, specs, bounds, solver)
    if not ok:
        hint = ""
        if shortcut and shortcut.get("R_needed_CMO") is not None:
            hint = (f" McCabe-Thiele (flujo molar constante) con estos parámetros estima que SÍ son alcanzables: "
                    f"R_min ≈ {shortcut['R_min_CMO']:.3f}, reflujo necesario con estos platos ≈ "
                    f"{shortcut['R_needed_CMO']:.3f}, N_min = {shortcut['N_min_total_reflux']}; el fallo es del "
                    "arranque numérico, no de la física: verifica en DWSIM o cambia platos/plato de alimentación")
        raise RuntimeError(f"ipopt no encontró un óptimo ({term}): las especificaciones pueden ser inalcanzables con "
                           f"{n_trays} platos dentro de los límites de reflujo/boil-up {bounds}.{hint}")
    start["smooth_eps_K"] = eps
    opt = _state(m)
    bal = component_balance(m)
    phantom = phantom_vapour(m)
    if any(abs(v) > BALANCE_TOL for v in bal.values()) or phantom > PHANTOM_VF_TOL:
        # M15b: never return numbers the seal cannot back. Near an azeotrope / pinch SmoothVLE's smoothing leaves vapour
        # in saturated liquids (≈ (eps/2)/(Tdew − Tbub)) and the total condenser drops it from the balance.
        gap = shortcut.get("dT_dew_bubble_distillate_K") if shortcut else None
        raise RuntimeError(
            f"la columna converge pero no se puede sellar: vapor ficticio {phantom:.2%} (tolerancia "
            f"{PHANTOM_VF_TOL:.1%}) y balance por componente {max(abs(v) for v in bal.values()):.1e} (tolerancia "
            f"{BALANCE_TOL:g}) con el suavizado de SmoothVLE en eps = {eps[0]:g} K" +
            (f"; el destilado pedido tiene T_rocío − T_burbuja = {gap:.3f} K" if gap is not None else "") +
            ". Pasa junto a un azeótropo o un pellizco: la formulación de equilibrio suavizado de esta tool no da "
            "duties ni composiciones confiables ahí. Relaja la pureza, o resuelve la columna en DWSIM "
            "(optimize_design_multi)")
    bub = NP.thermo_bubble_points(params, pressure_Pa, {s_: opt[s_]["mole_fractions"] for s_ in STREAMS})
    for s_ in STREAMS:
        bub["points"][s_]["T_idaes_K"] = opt[s_]["T_K"]
        bub["points"][s_]["dT_K"] = opt[s_]["T_K"] - bub["points"][s_]["T_bubble_K"]
    bub["tolerance_K"] = BUBBLE_T_TOL_K
    bub["verified"] = all(abs(v["dT_K"]) <= BUBBLE_T_TOL_K for v in bub["points"].values())
    T_lo, T_hi = min(opt[s_]["T_K"] for s_ in STREAMS), max(opt[s_]["T_K"] for s_ in STREAMS)
    warnings = NP.parameter_range_warnings(params, T_lo, T_hi) + cross["warnings"]
    out = {"components": components, "optimum": opt, "reboiler_duty_MW": opt["reboiler_duty_W"] / 1e6,
           "condenser_duty_MW": opt["condenser_duty_W"] / 1e6,
           "specs": [sp | {"value": _spec_value(opt, feed, sp)} for sp in specs],
           "duals_MW_per_unit_spec": duals, "active_specs": [k for k, v in duals.items() if v > 1e-6],
           "column": {"n_trays": n_trays, "feed_tray_from_top": feed_tray, "condenser": "total, at bubble point",
                      "reboiler": "partial", "pressure_Pa": pressure_Pa, "stages_incl_condenser_reboiler": n_trays + 2},
           "saturation_T_K_at_column_P": t_sat, "initial_point": start,
           "properties": {"package": "chemicals", "activity_model": "NRTL", "provenance": prov, "property_check": pcheck,
                          "nrtl_check": gcheck, "binary_parameters": params["pairs"],
                          "screening": {"B_P_over_RT_at_Tsat": virial, "max_B_P_RT": IP.MAX_B_P_RT,
                                        "azeotropes_with_these_parameters": azeos, "unifac_cross_check": cross},
                          "thermo_bubble_check": bub,
                          "excess_enthalpy": NP.excess_enthalpy_report(params, pressure_Pa, feed, opt, excess_enthalpy)},
           "shortcut": shortcut,
           "warnings": warnings,
           "engine": "IDAES TrayColumn (modular generic properties, fase líquida NRTL propia) + ipopt (idaes-ext)",
           "assumptions": [
               "Líquido NRTL (Raoult modificado, sin factor de Poynting) y vapor gas ideal (|B·P/RT| ≤ "
               f"{IP.MAX_B_P_RT}, Tsonopoulos sin términos polares; no capta la asociación en el vapor de ácidos "
               "carboxílicos). Parámetros binarios: una fuente etiquetada por par (properties.binary_parameters).",
               ("Entalpía del líquido = Σ x·h_i (paquete M8b) + h^E de NRTL con los mismos parámetros (Gibbs-Helmholtz); "
                "h^E no es un dato calorimétrico: con parámetros ajustados solo a VLE es una extrapolación del modelo "
                "(su peso en Q_reb está en properties.excess_enthalpy)." if excess_enthalpy else
                "Entalpía del líquido = Σ x·h_i (paquete M8b), SIN h^E (excess_enthalpy=False, pedido por el usuario)."),
               "Paquete puro generado de chemicals/thermo (fuentes en properties.provenance.sources), igual que M8b.",
               "Azeótropos buscados solo en pares binarios con estos parámetros; con 3+ compuestos y un azeótropo "
               "binario la tool se niega (no evalúa fronteras de destilación). UNIFAC Dortmund solo como comparación.",
               "Platos de equilibrio (eficiencia 100 %), sin caída de presión ni pérdidas de calor.",
               "Objetivo = carga del rehervidor; variables = reflujo y boil-up dentro de los límites dados (límites del "
               "solver, no datos del usuario)."]}
    if evidence:
        ev = _evidence(m, cfg, opt, duals, specs, bounds, solver, n_trays, feed_tray, feed, pressure_Pa, pcheck,
                       prov["temperature_window_K"], square_fd=square_route,
                       # multipliers taken from finite differences at DELTA_SPEC are re-checked with other steps
                       fd_deltas=(DELTA_SPEC / 4, DELTA_SPEC * 4) if fd_duals else (DELTA_SPEC, DELTA_SPEC / 4,
                                                                                     DELTA_SPEC * 4))
        ev["duals_source"] = duals_source
        ev["nrtl_verified"] = gcheck["verified"]
        ev["thermo_bubble_points_match"] = bub["verified"]
        ev["component_balance_rel"] = bal
        ev["component_balance_closed"] = all(abs(v) <= BALANCE_TOL for v in bal.values())
        ev["smooth_eps_K"] = eps
        ev["phantom_vapour_fraction"] = phantom
        ev["phantom_vapour_ok"] = phantom <= PHANTOM_VF_TOL
        if not ev["component_balance_closed"]:
            ev["reasons"].append(f"el balance por componente no cierra a {BALANCE_TOL:g}: {bal}")
        if not ev["phantom_vapour_ok"]:
            ev["reasons"].append(f"el suavizado de SmoothVLE (eps {eps[0]:g} K) deja {phantom:.2%} de vapor ficticio en "
                                 "líquidos saturados (zona de pellizco/azeótropo): duties y composiciones no confiables")
        if not gcheck["verified"]:
            ev["reasons"].append("γ/h^E del paquete IDAES no coinciden con thermo.nrtl / diferencia finita")
        if not bub["verified"]:
            ev["reasons"].append(f"T de producto vs T de burbuja de thermo difiere más de {BUBBLE_T_TOL_K} K")
        ev["verified"] = not ev["reasons"]
        out["evidence"] = ev
    out["seconds"] = time.perf_counter() - t0
    return out


START_POINTS = ((2.0, 2.0), (3.0, 3.0), (1.5, 3.0), (3.0, 2.0), (5.0, 5.0), (1.0, 1.5))
NRTL_START_POINTS = ((4.0, 0.2), (2.0, 0.1), (2.0, 0.2), (1.0, 0.1), (4.0, 0.1), (1.0, 0.2))


def _feed_two_phase_T(feed: dict, pressure_Pa: float, cfg_names: list[str], params: dict | None = None,
                      frac: float = 0.5) -> float:
    """T_bubble + frac·(T_dew − T_bubble) of the feed (thermo Psat; Raoult, or NRTL γ when params are given): a
    two-phase feed the IDAES TrayColumn initialises from (frac = 0.5, mid-way, in the ideal path)."""
    from scipy.optimize import brentq
    from packages.eo.ideal_properties import saturation_T
    from thermo import VaporPressure
    from chemicals import CAS_from_any
    ps = {c: VaporPressure(CASRN=CAS_from_any(c)) for c in cfg_names}
    ts = [saturation_T(c, pressure_Pa) for c in cfg_names]
    z = feed["z"]
    if params is None:
        lo, hi = min(ts) - 1.0, max(ts) + 1.0
        tb = brentq(lambda T: sum(z[c] * ps[c](T) for c in cfg_names) - pressure_Pa, lo, hi)
        td = brentq(lambda T: sum(z[c] / ps[c](T) for c in cfg_names) - 1.0 / pressure_Pa, lo, hi)
        return tb + frac * (td - tb)
    from packages.eo.nrtl_properties import gammas
    names = params["names"]
    lo, hi = min(ts) - 60.0, max(ts) + 60.0
    zs = [z[c] for c in names]
    tb = brentq(lambda T: sum(zi * g * ps[c](T) for zi, g, c in zip(zs, gammas(params, T, zs), names)) - pressure_Pa,
                lo, hi)

    def dew_resid(T):
        x = list(zs)
        for _ in range(200):
            g = gammas(params, T, x)
            xn = [zi * pressure_Pa / (gi * ps[c](T)) for zi, gi, c in zip(zs, g, names)]
            sx = sum(xn)
            xn = [v / sx for v in xn]
            if max(abs(a - b) for a, b in zip(x, xn)) < 1e-12:
                break
            x = xn
        return sx - 1.0
    td = brentq(dew_resid, lo, hi)
    return tb + frac * (td - tb)


def _square(m, solver) -> bool:
    from pyomo.environ import check_optimal_termination
    try:
        return bool(check_optimal_termination(solver.solve(m)))
    except Exception:  # noqa: BLE001
        return False


def _snapshot(m) -> list:
    from pyomo.environ import Var
    return [(v, v.value) for v in m.component_data_objects(Var, descend_into=True)]


def _restore(snap) -> None:
    for v, val in snap:
        v.set_value(val, skip_validation=True)


def _continue_feed_T(m, T_target: float, solver, max_steps: int = 200) -> bool:
    """Move the fixed feed temperature from its current value to T_target, halving the step on failure. A failed
    solve leaves the variables wherever ipopt stopped, so the last converged point is restored from a snapshot
    (re-solving from the failed iterate does not reliably return to it)."""
    from pyomo.environ import value
    u = m.fs.unit
    T = value(u.feed.temperature[0])
    step = (T_target - T) / 4.0
    n = 0
    snap = _snapshot(m)
    while abs(T_target - T) > 1e-9 and n < max_steps:
        n += 1
        T_try = T + step if abs(step) < abs(T_target - T) else T_target
        u.feed.temperature.fix(T_try)
        if _square(m, solver):
            T = T_try
            step *= 1.5
            snap = _snapshot(m)
        else:
            _restore(snap)
            u.feed.temperature.fix(T)
            step *= 0.5
            if abs(step) < 0.01:
                return False
    return abs(T_target - T) <= 1e-9


def _continue_scalar(m, solver, setter, start: float, target: float, min_step: float = 1e-3) -> float:
    """Homotopy on one scalar (setter(value) changes the model): tries the whole way first, halves the step on a
    failed square solve (restoring the last converged point) and doubles it on success. Returns the value reached."""
    lam, step, snap = 0.0, 1.0, _snapshot(m)
    while lam < 1.0:
        nxt = min(1.0, lam + step)
        setter(start + nxt * (target - start))
        if _square(m, solver):
            lam, snap, step = nxt, _snapshot(m), step * 2.0
        else:
            _restore(snap)
            step *= 0.5
            if step < min_step:
                setter(start + lam * (target - start))
                break
    return start + lam * (target - start)


def _spec_equalities(m, specs: list[dict], values: list[float]) -> None:
    """The specs as EQUALITIES with mutable targets, reflux and boil-up free: a square system when there are two specs."""
    from pyomo.environ import Constraint, Param
    u = m.fs.unit
    _drop_spec_equalities(m)
    for name in ("specs", "obj"):
        if hasattr(m, name):
            getattr(m, name).deactivate()
    m.spec_target = Param(range(len(specs)), mutable=True, initialize=dict(enumerate(values)))
    m.spec_eq = Constraint(range(len(specs)), rule=lambda mm, i: _spec_expr(mm, specs[i]) == mm.spec_target[i])
    u.condenser.reflux_ratio.unfix()
    u.reboiler.boilup_ratio.unfix()


def _drop_spec_equalities(m, reactivate: bool = True) -> None:
    for name in ("spec_eq", "spec_target"):
        if hasattr(m, name):
            m.del_component(name)
    if reactivate:
        for name in ("specs", "obj"):
            if hasattr(m, name):
                getattr(m, name).activate()


def _continue_specs(m, specs: list[dict], feed: dict, bounds: dict, solver) -> dict:
    """M15b start-up for two specs: from the converged square start (whatever RR/BR the IDAES initialisation accepted)
    the two spec values are moved, as equalities, from what that point gives to the requested minima. Every step is a
    square solve (Newton on a well-posed system) with step halving, instead of asking the NLP to cross the column from
    an infeasible point; the NLP then only polishes (and frees a spec that is not active at the optimum).
    Leaves the model at the point where both specs hold with equality, RR/BR fixed there. Returns what happened."""
    from pyomo.environ import value
    u = m.fs.unit
    snap0 = _snapshot(m)
    rr0, br0 = value(u.condenser.reflux_ratio), value(u.reboiler.boilup_ratio)
    st0 = _state(m)
    v0 = [_spec_value(st0, feed, sp) for sp in specs]
    v1 = [float(sp["min"]) for sp in specs]
    _spec_equalities(m, specs, v0)

    def setter(lam):
        for i in range(len(specs)):
            m.spec_target[i].set_value(v0[i] + lam * (v1[i] - v0[i]))
    ok = _square(m, solver)
    lam = _continue_scalar(m, solver, setter, 0.0, 1.0, min_step=SPEC_CONTINUATION_MIN_STEP) if ok else 0.0
    rr, br = value(u.condenser.reflux_ratio), value(u.reboiler.boilup_ratio)
    _drop_spec_equalities(m)
    inside = (bounds["reflux_ratio"][0] <= rr <= bounds["reflux_ratio"][1] and
              bounds["boilup_ratio"][0] <= br <= bounds["boilup_ratio"][1])
    info = {"from_spec_values": v0, "to_spec_values": v1, "fraction_reached": lam, "reflux_ratio": rr,
            "boilup_ratio": br, "inside_bounds": inside}
    if not (ok and lam >= 1.0 and inside):
        _restore(snap0)
        u.condenser.reflux_ratio.fix(rr0)
        u.reboiler.boilup_ratio.fix(br0)
        info["used"] = False
        return info
    u.condenser.reflux_ratio.fix(rr)
    u.reboiler.boilup_ratio.fix(br)
    info["used"] = True
    return info


def _square_at_specs(m, specs: list[dict], solver, targets: list[float] | None = None, reactivate: bool = True) -> bool:
    """Square solve with the specs held as equalities at `targets` (default: their minima); restores the point on
    failure."""
    snap = _snapshot(m)
    _spec_equalities(m, specs, targets if targets is not None else [float(sp["min"]) for sp in specs])
    ok = _square(m, solver)
    _drop_spec_equalities(m, reactivate)
    if not ok:
        _restore(snap)
    return ok


def _fd_duals_square(m, specs: list[dict], solver, delta: float = DELTA_SPEC) -> dict | None:
    """d(Q_reb, MW)/d(spec_i) by tightening one spec at a time on the square system (both specs active). With two
    active specs and two degrees of freedom these sensitivities ARE the Lagrange multipliers; both ≥ 0 ⇔ KKT point,
    and the critical cone is {0} (no free direction), so it is a strict local minimum."""
    from pyomo.environ import value
    u = m.fs.unit
    at = _snapshot(m)
    q0 = value(u.reboiler.heat_duty[0])
    mins = [float(sp["min"]) for sp in specs]
    out = {}
    for i, sp in enumerate(specs):
        t = list(mins)
        t[i] += delta
        ok = _square_at_specs(m, specs, solver, t, reactivate=False)
        if not ok:
            _restore(at)
            return None
        out[_spec_key(sp)] = (value(u.reboiler.heat_duty[0]) - q0) * 1e-6 / delta
        _restore(at)
    return out


def _solve_two_specs(m, specs: list[dict], feed: dict, bounds: dict, solver) -> dict:
    """M15b route for exactly two specs (a square system in RR and BR). The minimum-duty optimum of a column with a
    fixed number of trays sits next to minimum reflux, where ipopt's NLP is fragile (restoration failures from points a
    square Newton solve handles): so the NLP is not needed to GET there.
    1. _continue_specs: the spec values go, as equalities, from the converged start to the requested minima.
    2. SmoothVLE's eps goes down SMOOTH_EPS_STEPS on the square system, and on through SMOOTH_EPS_EXTRA only while the
       balance / phantom-vapour seal fails.
    3. Multipliers: ipopt's, from one NLP warm-started at that point (it also frees a spec that is not active at the
       optimum). If that NLP fails, the finite-difference sensitivities of the square system are the multipliers
       (see _fd_duals_square); a negative one means the active set is wrong → the route gives up (caller falls back).
    Returns {"ok", "termination", "duals", "eps", "info"}; on ok the model holds the optimum with m.specs/m.obj built."""
    from pyomo.environ import value
    info = _continue_specs(m, specs, feed, bounds, solver)
    out = {"ok": False, "termination": "la continuación en las specs no llegó", "duals": {}, "eps": [0.01, 0.0005],
           "info": info}
    if not info["used"]:
        return out
    reached = (0.01, 0.0005)
    for k, e1 in enumerate(SMOOTH_EPS_STEPS + SMOOTH_EPS_EXTRA):
        if k >= len(SMOOTH_EPS_STEPS) and _sealed(m):
            break
        e = (e1, e1 / 20)
        _set_smooth_eps(m, e)
        if _square_at_specs(m, specs, solver):
            reached = e
        else:
            _set_smooth_eps(m, reached)
            break
    info["eps_reached_square"] = list(reached)
    u = m.fs.unit
    at = _snapshot(m)
    q_sq = value(u.reboiler.heat_duty[0])
    ok, term, duals = _optimise(m, specs, bounds, solver)
    if ok and _sealed(m) and value(u.reboiler.heat_duty[0]) <= q_sq * (1 + 1e-6):
        info["duals_source"] = "multiplicadores de ipopt (NLP arrancado en la solución del sistema cuadrado)"
        info["nlp_moved_duty_rel"] = value(u.reboiler.heat_duty[0]) / q_sq - 1
        return {"ok": True, "termination": term, "duals": duals, "eps": list(reached), "info": info}
    _restore(at)
    for v, b in ((u.condenser.reflux_ratio, bounds["reflux_ratio"]), (u.reboiler.boilup_ratio, bounds["boilup_ratio"])):
        v.unfix()
        v.setlb(b[0])
        v.setub(b[1])
    fd = _fd_duals_square(m, specs, solver)
    info["nlp_at_final_eps"] = term
    if fd is None or any(v < -1e-9 for v in fd.values()):
        info["fd_duals"] = fd
        out["termination"] = f"NLP en el eps final: {term}; sensibilidades del sistema cuadrado no válidas"
        return out
    info["duals_source"] = ("sensibilidades por diferencia finita del sistema cuadrado con las dos specs activas (= "
                            "multiplicadores; el NLP de ipopt no convergió desde ese punto: " + term + ")")
    return {"ok": True, "termination": "optimal (sistema cuadrado, specs activas; KKT por sensibilidades ≥ 0)",
            "duals": {k: max(v, 0.0) for k, v in fd.items()}, "eps": list(reached), "info": info}


def _hold_distillate(m, D: float) -> None:
    """Distillate flow (an Expression of the condenser split) held by a constraint; the boil-up ratio is freed."""
    from pyomo.environ import Constraint
    if hasattr(m, "hold_distillate"):
        m.del_component(m.hold_distillate)
    m.hold_distillate = Constraint(expr=m.fs.unit.condenser.distillate.flow_mol[0] == D)
    m.fs.unit.reboiler.boilup_ratio.unfix()


def _spec_start_points(feed: dict, specs: list[dict], rr_list=(2.0, 4.0)) -> list[tuple[float, float]]:
    """(RR, BR) pairs whose distillate/feed ratio matches the specs: D/F ≈ z_c·recovery/purity for a component c
    specified in the distillate (B/F likewise for the bottoms); BR = (RR + 1)·D/B from the column balance."""
    z = feed["z"]
    frac = {}
    for sp in specs:
        frac.setdefault((sp["stream"], sp["component"]), {})[sp["kind"]] = sp["min"]
    ratios = []
    for (stream, c), k in frac.items():
        f = z[c] * k.get("recovery", 1.0) / k.get("mole_fraction", 1.0) if "mole_fraction" in k else z[c] * k["recovery"]
        f = min(max(f, 1e-3), 0.999)
        ratios.append(f if stream == "distillate" else 1.0 - f)
    out = []
    for dF in sorted(set(round(r, 6) for r in ratios)):
        for rr in rr_list:
            out.append((rr, (rr + 1.0) * dF / (1.0 - dF)))
    return out


def _initial_point(n_trays, feed_tray, feed, pressure_Pa, cfg, bounds, solver, first=None, params=None,
                   extra_starts=(), homotopy_starts=()):
    """Square solve at the first (RR, BR) of the list that converges (the column is a hard NLP from a cold start).
    IDAES's TrayColumn does not start from a subcooled/superheated feed (M8a's package fails too, e.g. benzene-toluene
    at 340-362 K), so on failure the column is started with a two-phase feed and the feed temperature is continued to
    the user's value."""
    import idaes.logger as idaeslog
    clip = lambda v, b: min(max(v, b[0]), b[1])  # noqa: E731
    comps = list(feed["z"]) if feed.get("z") else list(COMPONENTS)
    tried = []
    # NRTL path: start from a feed only slightly vaporised (30 % / 10 % of the way from bubble to dew), so that a low
    # boil-up gives a distillate of the order the specs need; the ideal path keeps M8b's mid-way start.
    modes = ("direct", "feed_T_continuation") if params is None else ("two_phase_0.3", "two_phase_0.1", "direct")
    if params is None:
        starts = ((first,) if first else ()) + tuple(extra_starts) + START_POINTS
        plan = [(s_, mode) for s_ in starts for mode in modes]
    else:       # low boil-up first: what converged on the methanol-water and ethanol-water grids (M15)
        starts = ((first,) if first else ()) + NRTL_START_POINTS + tuple(extra_starts) + START_POINTS
        plan = [(s_, mode) for s_ in starts for mode in modes]
        # M15b: after the first M15 attempt, the non-ideality homotopy (ideal liquid → NRTL) at the McCabe-Thiele
        # starts (a distillate rate the specs imply, away from the azeotrope's pinch) and at the first M15 pair
        homo = [(s_, "ideal_homotopy") for s_ in tuple(homotopy_starts)[:2] + (starts[0],)]
        plan = plan[:1] + homo + plan[1:]
    for (rr, br), mode in plan:
        rr, br = clip(rr, bounds["reflux_ratio"]), clip(br, bounds["boilup_ratio"])
        if True:
            f0 = dict(feed)
            if mode == "ideal_homotopy":
                f0["T_K"] = _feed_two_phase_T(feed, pressure_Pa, comps, None, 0.3)       # Raoult bubble/dew
            elif mode != "direct":
                frac = 0.5 if mode == "feed_T_continuation" else float(mode.split("_")[-1])
                f0["T_K"] = _feed_two_phase_T(feed, pressure_Pa, comps, params, frac)
            m = build_column(n_trays, feed_tray, f0, pressure_Pa, config=cfg)
            m.fs.unit.condenser.reflux_ratio.fix(rr)
            m.fs.unit.reboiler.boilup_ratio.fix(br)
            if mode == "ideal_homotopy":
                m.fs.properties.Liq.nrtl_scale.set_value(0.0)
            try:
                m.fs.unit.initialize(outlvl=idaeslog.CRITICAL)
            except Exception:  # noqa: BLE001
                pass
            ok = _square(m, solver)
            if mode == "ideal_homotopy":
                # λ: 0 → 1 scales every τ (ideal liquid → NRTL) while the feed goes from 30 % vaporised under Raoult
                # to 30 % vaporised under NRTL, so the vapour traffic of the column stays comparable along the path
                scale = m.fs.properties.Liq.nrtl_scale
                T_id, T_nr = f0["T_K"], _feed_two_phase_T(feed, pressure_Pa, comps, params, 0.3)

                def _homotopy(lam, scale=scale, T_id=T_id, T_nr=T_nr, m=m):
                    scale.set_value(lam)
                    m.fs.unit.feed.temperature.fix(T_id + lam * (T_nr - T_id))
                ok = ok and _continue_scalar(m, solver, _homotopy, 0.0, 1.0, min_step=1e-3) >= 1.0
                scale.set_value(1.0)
            if ok and mode != "direct":
                if params is not None:
                    # NRTL path: hold the distillate flow and free the boil-up while the feed is cooled (a subcooled
                    # feed condenses vapour; at fixed boil-up the distillate can collapse), then fix the boil-up again
                    from pyomo.environ import value
                    u = m.fs.unit
                    _hold_distillate(m, value(u.condenser.distillate.flow_mol[0]))
                    ok = _continue_feed_T(m, feed["T_K"], solver)
                    br_now = value(u.reboiler.boilup_ratio)
                    m.del_component(m.hold_distillate)
                    u.reboiler.boilup_ratio.fix(br_now)
                    ok = ok and bounds["boilup_ratio"][0] <= br_now <= bounds["boilup_ratio"][1] and _square(m, solver)
                    br = br_now
                else:
                    ok = _continue_feed_T(m, feed["T_K"], solver)
            if ok:
                info = {"reflux_ratio": rr, "boilup_ratio": br, "mode": mode, "start_feed_T_K": f0["T_K"],
                        "tried": tried}
                return m, info
            tried.append((rr, br, mode))
    raise RuntimeError(f"la columna no converge en ningún punto inicial {tried}; revisa alimentación, presión y número "
                       "de platos")


def _meets(state: dict, feed: dict, specs: list[dict]) -> bool:
    return all(_spec_value(state, feed, sp) >= sp["min"] - SPEC_TOL for sp in specs)


def _evidence(m, cfg, opt, duals, specs, bounds, solver, n_trays, feed_tray, feed, pressure_Pa, pcheck, window,
              square_fd: bool = False, fd_deltas: tuple = (DELTA_SPEC, DELTA_SPEC / 4, DELTA_SPEC * 4)) -> dict:
    from pyomo.environ import check_optimal_termination, value
    reasons = []
    specs_met = _meets(opt, feed, specs)
    if not specs_met:
        reasons.append("el óptimo no cumple las especificaciones")
    props_ok = pcheck is None or pcheck["verified"]
    if not props_ok:
        bad = [k for k, v in pcheck["within_tolerance"].items() if not v]
        reasons.append(f"propiedades generadas fuera de tolerancia vs chemicals/DIPPR: {bad}")
    lo, hi = window
    inside = all(lo + 1.0 < opt[s]["T_K"] < hi - 1.0 for s in STREAMS)
    if not inside:
        reasons.append(f"temperatura de producto en el borde de la ventana del paquete ({lo:.0f}-{hi:.0f} K)")
    # local optimality: neighbours of (RR, BR)
    u = m.fs.unit
    rr0, br0, q0 = opt["reflux_ratio"], opt["boilup_ratio"], opt["reboiler_duty_W"]
    m.specs.deactivate()
    m.obj.deactivate()
    neighbours, better_feasible = [], []
    at_opt = _snapshot(m)
    for fr in (1 - PERTURB, 1.0, 1 + PERTURB):
        for fb in (1 - PERTURB, 1.0, 1 + PERTURB):
            if fr == fb == 1.0:
                continue
            _restore(at_opt)              # every neighbour starts from the optimum
            u.condenser.reflux_ratio.fix(rr0 * fr)
            u.reboiler.boilup_ratio.fix(br0 * fb)
            try:
                res = solver.solve(m)
                solved = bool(check_optimal_termination(res))
            except (ValueError, RuntimeError):      # ipopt evaluation error → Pyomo cannot load the result
                solved = False
            if not solved:
                neighbours.append({"rr": rr0 * fr, "br": br0 * fb, "solved": False})
                continue
            s = _state(m)
            feas = _meets(s, feed, specs)
            neighbours.append({"rr": rr0 * fr, "br": br0 * fb, "solved": True, "meets_specs": feas,
                               "reboiler_duty_W": s["reboiler_duty_W"]})
            if feas and s["reboiler_duty_W"] < q0 * (1 - 1e-9):
                better_feasible.append(neighbours[-1])
    local_opt = not better_feasible and all(n["solved"] for n in neighbours)
    if not local_opt:
        reasons.append("un vecino cumple las especificaciones con menos carga, o no resolvió: el punto no es un óptimo local")
    # multipliers vs finite differences: re-optimise with each active spec tightened, warm-started from the optimum
    dual_checks = {}
    for i, sp in enumerate(specs):
        key = _spec_key(sp)
        if duals[key] <= 1e-6:
            dual_checks[key] = {"active": False}
            continue
        ok, fd, used = False, float("nan"), None
        for delta in fd_deltas:   # a failed re-optimisation retries another step
            tight = [dict(s2) for s2 in specs]
            tight[i]["min"] = sp["min"] + delta
            _restore(at_opt)
            u.condenser.reflux_ratio.fix(rr0)
            u.reboiler.boilup_ratio.fix(br0)
            try:
                if square_fd and all(v > 1e-6 for v in duals.values()):
                    # M15b: both specs active → the tightened optimum is the square solution at the tightened spec
                    for v_ in (u.condenser.reflux_ratio, u.reboiler.boilup_ratio):
                        v_.unfix()
                    ok = _square_at_specs(m, tight, solver, reactivate=False)
                else:
                    ok, _, _ = _optimise(m, tight, bounds, solver)
            except (ValueError, RuntimeError):
                ok = False
            if ok:
                fd, used = (value(u.reboiler.heat_duty[0]) * 1e-6 - q0 * 1e-6) / delta, delta
                break
        rel = abs(fd - duals[key]) / max(abs(duals[key]), 1e-12) if ok else float("inf")
        dual_checks[key] = {"active": True, "dual": duals[key], "finite_difference": fd, "delta_spec": used, "rel_diff": rel,
                            "consistent": ok and rel < DUAL_REL_TOL}
        if not dual_checks[key]["consistent"]:
            reasons.append(f"el multiplicador de {key} no coincide con la diferencia finita ({rel:.1%})")
    # restore the optimum on the model
    _restore(at_opt)
    u.condenser.reflux_ratio.fix(rr0)
    u.reboiler.boilup_ratio.fix(br0)
    _square(m, solver)
    return {"ipopt_optimal": True, "specs_met": specs_met, "properties_verified": props_ok,
            "inside_property_window": inside, "local_optimum": local_opt, "neighbours": neighbours,
            "duals_consistent": all(d.get("consistent", True) for d in dual_checks.values()), "dual_checks": dual_checks,
            "tolerances": {"spec": SPEC_TOL, "perturbation": PERTURB, "delta_spec": DELTA_SPEC, "dual_rel": DUAL_REL_TOL},
            "verified": not reasons, "reasons": reasons}
