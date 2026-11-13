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


def build_column(n_trays: int, feed_tray: int, feed: dict, pressure_Pa: float, corrected: bool = True, config: dict | None = None):
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
    res = solver.solve(m)
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
                       property_package: str = "chemicals") -> dict:
    """Minimum reboiler duty of a tray column meeting purity/recovery specs, solved as one NLP (IDAES + ipopt), for an
    ideal mixture whose property package is generated from chemicals/thermo. spec = {"stream": "distillate"|"bottoms",
    "component": name, "kind": "mole_fraction"|"recovery", "min": value}."""
    from packages.eo import ideal_properties as IP
    _validate(components, feed_flow_mol_s, feed_T_K, feed_P_Pa, feed_mole_fractions, n_trays, feed_tray, pressure_Pa,
              specs, property_package)
    components = list(components)
    t0 = time.perf_counter()
    ideality = IP.ideality_check(components, pressure_Pa)
    if not ideality["ideal"]:
        raise ValueError("mezcla no ideal: esta tool solo resuelve VLE ideal (Raoult + gas ideal). "
                         + "; ".join(ideality["reasons"]) + ". Usa la columna de DWSIM con NRTL/UNIQUAC/UNIFAC "
                         "(optimize_design_multi).")
    t_sat = {c: IP.saturation_T(c, pressure_Pa) for c in components}
    if property_package == "chemicals":
        cfg, prov = IP.build_configuration(components, pressure_Pa, extra_T=[feed_T_K])
        lo, hi = min(t_sat.values()), max(t_sat.values())
        pcheck = IP.property_check(cfg, components, [lo + f * (hi - lo) for f in (0.0, 1 / 3, 2 / 3, 1.0)])
    else:
        cfg, pcheck = bt_ideal_configuration(True), None
        prov = {"package": "IDAES BT_ideal (Perry's/RPP4) con cp líquido corregido (M8a)", "temperature_window_K": (273.15, 450.0)}
    feed = {"flow_mol_s": feed_flow_mol_s, "T_K": feed_T_K, "P_Pa": feed_P_Pa, "z": dict(feed_mole_fractions)}
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


START_POINTS = ((2.0, 2.0), (3.0, 3.0), (1.5, 3.0), (3.0, 2.0), (5.0, 5.0), (1.0, 1.5))


def _feed_two_phase_T(feed: dict, pressure_Pa: float, cfg_names: list[str]) -> float:
    """Mid-way between the Raoult bubble and dew temperature of the feed (thermo Psat): a two-phase feed the IDAES
    TrayColumn initialises from."""
    from scipy.optimize import brentq
    from packages.eo.ideal_properties import saturation_T
    from thermo import VaporPressure
    from chemicals import CAS_from_any
    ps = {c: VaporPressure(CASRN=CAS_from_any(c)) for c in cfg_names}
    ts = [saturation_T(c, pressure_Pa) for c in cfg_names]
    lo, hi = min(ts) - 1.0, max(ts) + 1.0
    z = feed["z"]
    tb = brentq(lambda T: sum(z[c] * ps[c](T) for c in cfg_names) - pressure_Pa, lo, hi)
    td = brentq(lambda T: sum(z[c] / ps[c](T) for c in cfg_names) - 1.0 / pressure_Pa, lo, hi)
    return 0.5 * (tb + td)


def _square(m, solver) -> bool:
    from pyomo.environ import check_optimal_termination
    try:
        return bool(check_optimal_termination(solver.solve(m)))
    except Exception:  # noqa: BLE001
        return False


def _continue_feed_T(m, T_target: float, solver, max_steps: int = 200) -> bool:
    """Move the fixed feed temperature from its current value to T_target, halving the step on failure."""
    from pyomo.environ import value
    u = m.fs.unit
    T = value(u.feed.temperature[0])
    step = (T_target - T) / 4.0
    n = 0
    while abs(T_target - T) > 1e-9 and n < max_steps:
        n += 1
        T_try = T + step if abs(step) < abs(T_target - T) else T_target
        u.feed.temperature.fix(T_try)
        if _square(m, solver):
            T = T_try
            step *= 1.5
        else:
            u.feed.temperature.fix(T)
            _square(m, solver)
            step *= 0.5
            if abs(step) < 0.01:
                return False
    return abs(T_target - T) <= 1e-9


def _initial_point(n_trays, feed_tray, feed, pressure_Pa, cfg, bounds, solver, first=None):
    """Square solve at the first (RR, BR) of the list that converges (the column is a hard NLP from a cold start).
    IDAES's TrayColumn does not start from a subcooled/superheated feed (M8a's package fails too, e.g. benzene-toluene
    at 340-362 K), so on failure the column is started with a two-phase feed and the feed temperature is continued to
    the user's value."""
    import idaes.logger as idaeslog
    clip = lambda v, b: min(max(v, b[0]), b[1])  # noqa: E731
    comps = list(feed["z"]) if feed.get("z") else list(COMPONENTS)
    tried = []
    for rr, br in ((first,) if first else ()) + START_POINTS:
        rr, br = clip(rr, bounds["reflux_ratio"]), clip(br, bounds["boilup_ratio"])
        for mode in ("direct", "feed_T_continuation"):
            f0 = dict(feed)
            if mode == "feed_T_continuation":
                f0["T_K"] = _feed_two_phase_T(feed, pressure_Pa, comps)
            m = build_column(n_trays, feed_tray, f0, pressure_Pa, config=cfg)
            m.fs.unit.condenser.reflux_ratio.fix(rr)
            m.fs.unit.reboiler.boilup_ratio.fix(br)
            try:
                m.fs.unit.initialize(outlvl=idaeslog.CRITICAL)
            except Exception:  # noqa: BLE001
                pass
            ok = _square(m, solver)
            if ok and mode == "feed_T_continuation":
                ok = _continue_feed_T(m, feed["T_K"], solver)
            if ok:
                return m, {"reflux_ratio": rr, "boilup_ratio": br, "mode": mode,
                           "start_feed_T_K": f0["T_K"], "tried": tried}
            tried.append((rr, br, mode))
    raise RuntimeError(f"la columna no converge en ningún punto inicial {tried}; revisa alimentación, presión y número "
                       "de platos")


def _meets(state: dict, feed: dict, specs: list[dict]) -> bool:
    return all(_spec_value(state, feed, sp) >= sp["min"] - SPEC_TOL for sp in specs)


def _evidence(m, cfg, opt, duals, specs, bounds, solver, n_trays, feed_tray, feed, pressure_Pa, pcheck, window) -> dict:
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
    for fr in (1 - PERTURB, 1.0, 1 + PERTURB):
        for fb in (1 - PERTURB, 1.0, 1 + PERTURB):
            if fr == fb == 1.0:
                continue
            u.condenser.reflux_ratio.fix(rr0 * fr)
            u.reboiler.boilup_ratio.fix(br0 * fb)
            res = solver.solve(m)
            if not check_optimal_termination(res):
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
        tight = [dict(s2) for s2 in specs]
        tight[i]["min"] = sp["min"] + DELTA_SPEC
        u.condenser.reflux_ratio.fix(rr0)
        u.reboiler.boilup_ratio.fix(br0)
        _square(m, solver)
        ok, _, _ = _optimise(m, tight, bounds, solver)
        fd = (value(u.reboiler.heat_duty[0]) * 1e-6 - q0 * 1e-6) / DELTA_SPEC if ok else float("nan")
        rel = abs(fd - duals[key]) / max(abs(duals[key]), 1e-12) if ok else float("inf")
        dual_checks[key] = {"active": True, "dual": duals[key], "finite_difference": fd, "rel_diff": rel,
                            "consistent": ok and rel < DUAL_REL_TOL}
        if not dual_checks[key]["consistent"]:
            reasons.append(f"el multiplicador de {key} no coincide con la diferencia finita ({rel:.1%})")
    # restore the optimum on the model
    u.condenser.reflux_ratio.fix(rr0)
    u.reboiler.boilup_ratio.fix(br0)
    _square(m, solver)
    return {"ipopt_optimal": True, "specs_met": specs_met, "properties_verified": props_ok,
            "inside_property_window": inside, "local_optimum": local_opt, "neighbours": neighbours,
            "duals_consistent": all(d.get("consistent", True) for d in dual_checks.values()), "dual_checks": dual_checks,
            "tolerances": {"spec": SPEC_TOL, "perturbation": PERTURB, "delta_spec": DELTA_SPEC, "dual_rel": DUAL_REL_TOL},
            "verified": not reasons, "reasons": reasons}
