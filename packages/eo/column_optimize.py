"""Equation-oriented distillation optimisation (bloque M8a of docs/MAPA_MOTORES.md): IDAES TrayColumn solved
simultaneously by ipopt (idaes-ext 3.4.2 binaries, installed with `idaes get-extensions`, EPL-2.0 + HSL for IDAES).

DWSIM solves a column sequentially: an optimiser around it needs one full column solve per trial (optimize_design_multi).
Here the whole column (MESH equations on every tray, condenser, reboiler) and the purity constraints are ONE nonlinear
program, so ipopt returns the minimum reboiler duty, the optimal reflux and boil-up ratios AND the Lagrange multipliers:
how many MW each extra unit of purity costs.

Systems: only the property configurations IDAES ships and tests (benzene-toluene, ideal VLE `BT_ideal`, Perry's/RPP4
parameters). Other compounds need a verified configuration first (M8b).

IDAES is BSD and runs in this process; ipopt runs as a separate executable launched by Pyomo, with max_cpu_time.

Evidence → seal (`verified` only when all hold):
  * ipopt_optimal: termination condition optimal
  * specs_met: each purity ≥ its spec − SPEC_TOL
  * local_optimum: square solves at the 8 neighbours (reflux, boil-up) × (1 ± PERTURB): every neighbour that meets
    the specs has a duty ≥ the optimum (− 1e-9 relative)
  * duals_consistent: for each active spec, re-optimising with the spec tightened by DELTA_SPEC moves the duty by the
    multiplier × DELTA_SPEC within DUAL_REL_TOL
The reference case of the IDAES test suite (TestBTXIdealGeneric) is reproduced in tests/test_eo_column.py.
"""
from __future__ import annotations

import math
import time

SPEC_TOL = 1e-6
PERTURB = 0.01
DELTA_SPEC = 1e-4
DUAL_REL_TOL = 0.05
MAX_CPU_S = 120
SYSTEMS = ("benzene-toluene ideal",)
COMPONENTS = ("benzene", "toluene")


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
    return get_solver("ipopt", options={"max_cpu_time": MAX_CPU_S})


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


def build_column(n_trays: int, feed_tray: int, feed: dict, pressure_Pa: float, corrected: bool = True):
    """IDAES TrayColumn (total condenser at bubble point, partial reboiler), benzene-toluene ideal."""
    from pyomo.environ import ConcreteModel
    from idaes.core import FlowsheetBlock
    from idaes.models_extra.column_models import TrayColumn
    from idaes.models_extra.column_models.condenser import CondenserType, TemperatureSpec
    from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = GenericParameterBlock(**bt_ideal_configuration(corrected))
    m.fs.unit = TrayColumn(number_of_trays=n_trays, feed_tray_location=feed_tray, condenser_type=CondenserType.totalCondenser,
                           condenser_temperature_spec=TemperatureSpec.atBubblePoint, property_package=m.fs.properties,
                           has_heat_transfer=False, has_pressure_change=False)
    u = m.fs.unit
    u.feed.flow_mol.fix(feed["flow_mol_s"])
    u.feed.temperature.fix(feed["T_K"])
    u.feed.pressure.fix(feed["P_Pa"])
    u.feed.mole_frac_comp[0, "benzene"].fix(feed["x_benzene"])
    u.feed.mole_frac_comp[0, "toluene"].fix(1.0 - feed["x_benzene"])
    u.condenser.condenser_pressure.fix(pressure_Pa)
    return m


def _state(m) -> dict:
    from pyomo.environ import value
    u = m.fs.unit
    d, b = u.condenser.distillate, u.reboiler.bottoms
    return {"reflux_ratio": value(u.condenser.reflux_ratio), "boilup_ratio": value(u.reboiler.boilup_ratio),
            "reboiler_duty_W": value(u.reboiler.heat_duty[0]), "condenser_duty_W": value(u.condenser.heat_duty[0]),
            "distillate": {"flow_mol_s": value(d.flow_mol[0]), "x_benzene": value(d.mole_frac_comp[0, "benzene"]),
                           "T_K": value(d.temperature[0])},
            "bottoms": {"flow_mol_s": value(b.flow_mol[0]), "x_toluene": value(b.mole_frac_comp[0, "toluene"]),
                        "T_K": value(b.temperature[0])}}


def simulate_column(n_trays: int, feed_tray: int, feed: dict, pressure_Pa: float, reflux_ratio: float,
                    boilup_ratio: float, corrected: bool = True) -> dict:
    """Square solve (no optimisation). corrected=False is IDAES's own configuration (its test-suite reference)."""
    from pyomo.environ import check_optimal_termination
    import idaes.logger as idaeslog
    m = build_column(n_trays, feed_tray, feed, pressure_Pa, corrected)
    m.fs.unit.condenser.reflux_ratio.fix(reflux_ratio)
    m.fs.unit.reboiler.boilup_ratio.fix(boilup_ratio)
    m.fs.unit.initialize(outlvl=idaeslog.CRITICAL)
    res = _solver().solve(m)
    if not check_optimal_termination(res):
        raise RuntimeError(f"ipopt no resolvió la columna: {res.solver.termination_condition}")
    return _state(m)


def _optimise(m, xd_min: float, xb_min: float, bounds: dict, solver):
    from pyomo.environ import Constraint, Objective, Suffix, minimize, check_optimal_termination
    u = m.fs.unit
    for v, (lo, hi) in ((u.condenser.reflux_ratio, bounds["reflux_ratio"]), (u.reboiler.boilup_ratio, bounds["boilup_ratio"])):
        v.unfix()
        v.setlb(lo)
        v.setub(hi)
    for name in ("spec_xd", "spec_xb", "obj", "dual"):
        if hasattr(m, name):
            m.del_component(name)
    m.spec_xd = Constraint(expr=u.condenser.distillate.mole_frac_comp[0, "benzene"] >= xd_min)
    m.spec_xb = Constraint(expr=u.reboiler.bottoms.mole_frac_comp[0, "toluene"] >= xb_min)
    m.obj = Objective(expr=u.reboiler.heat_duty[0] * 1e-6, sense=minimize)
    m.dual = Suffix(direction=Suffix.IMPORT)
    res = solver.solve(m)
    ok = check_optimal_termination(res)
    # ipopt's multiplier for g(x) ≥ b is ≥ 0 in Pyomo's sign convention for a minimisation: d(obj)/d(b)
    duals = {"x_benzene_distillate": abs(m.dual.get(m.spec_xd, 0.0)), "x_toluene_bottoms": abs(m.dual.get(m.spec_xb, 0.0))}
    return ok, str(res.solver.termination_condition), duals


def eo_column_optimize(system: str, feed_flow_mol_s: float, feed_T_K: float, feed_P_Pa: float, feed_x_benzene: float,
                       n_trays: int, feed_tray: int, pressure_Pa: float, x_benzene_distillate_min: float,
                       x_toluene_bottoms_min: float, reflux_ratio_bounds: tuple[float, float] = (0.1, 20.0),
                       boilup_ratio_bounds: tuple[float, float] = (0.1, 20.0), evidence: bool = True) -> dict:
    """Minimum reboiler duty of a tray column meeting two purity specs, solved as one NLP (IDAES + ipopt)."""
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}: IDAES only ships a verified configuration for these; "
                         "other compounds are bloque M8b")
    if not (0 < feed_x_benzene < 1 and 0 < x_benzene_distillate_min < 1 and 0 < x_toluene_bottoms_min < 1):
        raise ValueError("mole fractions must be in (0, 1)")
    if not (1 <= feed_tray <= n_trays) or n_trays < 2:
        raise ValueError("feed_tray must be between 1 (top tray) and n_trays")
    for k, v in dict(feed_flow_mol_s=feed_flow_mol_s, feed_T_K=feed_T_K, feed_P_Pa=feed_P_Pa, pressure_Pa=pressure_Pa).items():
        if v is None or v <= 0:
            raise ValueError(f"{k} must be positive")
    import idaes.logger as idaeslog
    from pyomo.environ import check_optimal_termination
    t0 = time.perf_counter()
    feed = {"flow_mol_s": feed_flow_mol_s, "T_K": feed_T_K, "P_Pa": feed_P_Pa, "x_benzene": feed_x_benzene}
    bounds = {"reflux_ratio": tuple(reflux_ratio_bounds), "boilup_ratio": tuple(boilup_ratio_bounds)}
    solver = _solver()
    m = build_column(n_trays, feed_tray, feed, pressure_Pa)
    u = m.fs.unit
    # initial point: a mid-range square solve
    u.condenser.reflux_ratio.fix(min(max(2.0, bounds["reflux_ratio"][0]), bounds["reflux_ratio"][1]))
    u.reboiler.boilup_ratio.fix(min(max(2.0, bounds["boilup_ratio"][0]), bounds["boilup_ratio"][1]))
    u.initialize(outlvl=idaeslog.CRITICAL)
    if not check_optimal_termination(solver.solve(m)):
        raise RuntimeError("la columna no converge en el punto inicial (RR = BR = 2); revisa alimentación y presión")
    ok, term, duals = _optimise(m, x_benzene_distillate_min, x_toluene_bottoms_min, bounds, solver)
    if not ok:
        raise RuntimeError(f"ipopt no encontró un óptimo ({term}): las purezas pueden ser inalcanzables con {n_trays} "
                           f"platos dentro de los límites de reflujo/boil-up {bounds}")
    opt = _state(m)
    out = {"system": system, "optimum": opt, "reboiler_duty_MW": opt["reboiler_duty_W"] / 1e6,
           "condenser_duty_MW": opt["condenser_duty_W"] / 1e6,
           "duals_MW_per_unit_purity": duals,
           "active_specs": [k for k, v in duals.items() if v > 1e-6],
           "column": {"n_trays": n_trays, "feed_tray_from_top": feed_tray, "condenser": "total, at bubble point",
                      "reboiler": "partial", "pressure_Pa": pressure_Pa, "stages_incl_condenser_reboiler": n_trays + 2},
           "engine": "IDAES TrayColumn (BT_ideal, modular properties) + ipopt (idaes-ext)",
           "assumptions": ["VLE ideal (Raoult) con Psat y entalpías de la configuración BT_ideal de IDAES (Perry's / RPP4), "
                           "con el cp líquido corregido (IDAES lo declara en J/kmol·K: 1000× menor; ΔHvap salía +26-43 %).",
                           "Platos de equilibrio (eficiencia 100 %), sin caída de presión ni pérdidas de calor.",
                           "Objetivo = carga del rehervidor; variables = reflujo y boil-up dentro de los límites dados."]}
    if evidence:
        out["evidence"] = _evidence(m, opt, duals, x_benzene_distillate_min, x_toluene_bottoms_min, bounds, solver,
                                    n_trays, feed_tray, feed, pressure_Pa)
    out["seconds"] = time.perf_counter() - t0
    return out


def _evidence(m, opt, duals, xd_min, xb_min, bounds, solver, n_trays, feed_tray, feed, pressure_Pa) -> dict:
    from pyomo.environ import check_optimal_termination, value
    import idaes.logger as idaeslog
    reasons = []
    specs_met = (opt["distillate"]["x_benzene"] >= xd_min - SPEC_TOL) and (opt["bottoms"]["x_toluene"] >= xb_min - SPEC_TOL)
    if not specs_met:
        reasons.append("el óptimo no cumple las purezas")
    # local optimality: neighbours of (RR, BR)
    u = m.fs.unit
    rr0, br0, q0 = opt["reflux_ratio"], opt["boilup_ratio"], opt["reboiler_duty_W"]
    m.spec_xd.deactivate()
    m.spec_xb.deactivate()
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
            feas = s["distillate"]["x_benzene"] >= xd_min - SPEC_TOL and s["bottoms"]["x_toluene"] >= xb_min - SPEC_TOL
            neighbours.append({"rr": rr0 * fr, "br": br0 * fb, "solved": True, "meets_specs": feas,
                               "reboiler_duty_W": s["reboiler_duty_W"]})
            if feas and s["reboiler_duty_W"] < q0 * (1 - 1e-9):
                better_feasible.append(neighbours[-1])
    local_opt = not better_feasible and all(n["solved"] for n in neighbours)
    if not local_opt:
        reasons.append("un vecino cumple las purezas con menos carga, o no resolvió: el punto no es un óptimo local")
    # multipliers vs finite differences (re-optimise with each active spec tightened)
    dual_checks = {}
    for key, xd, xb in (("x_benzene_distillate", xd_min + DELTA_SPEC, xb_min), ("x_toluene_bottoms", xd_min, xb_min + DELTA_SPEC)):
        if duals[key] <= 1e-6:
            dual_checks[key] = {"active": False}
            continue
        mm = build_column(n_trays, feed_tray, feed, pressure_Pa)
        mm.fs.unit.condenser.reflux_ratio.fix(rr0)
        mm.fs.unit.reboiler.boilup_ratio.fix(br0)
        mm.fs.unit.initialize(outlvl=idaeslog.CRITICAL)
        solver.solve(mm)
        ok, _, _ = _optimise(mm, xd, xb, bounds, solver)
        fd = (value(mm.fs.unit.reboiler.heat_duty[0]) * 1e-6 - q0 * 1e-6) / DELTA_SPEC if ok else float("nan")
        rel = abs(fd - duals[key]) / max(abs(duals[key]), 1e-12) if ok else float("inf")
        dual_checks[key] = {"active": True, "dual": duals[key], "finite_difference": fd, "rel_diff": rel,
                            "consistent": ok and rel < DUAL_REL_TOL}
        if not dual_checks[key]["consistent"]:
            reasons.append(f"el multiplicador de {key} no coincide con la diferencia finita ({rel:.1%})")
    # restore the optimum on the model
    u.condenser.reflux_ratio.fix(rr0)
    u.reboiler.boilup_ratio.fix(br0)
    return {"ipopt_optimal": True, "specs_met": specs_met, "local_optimum": local_opt, "neighbours": neighbours,
            "duals_consistent": all(d.get("consistent", True) for d in dual_checks.values()), "dual_checks": dual_checks,
            "tolerances": {"spec": SPEC_TOL, "perturbation": PERTURB, "delta_spec": DELTA_SPEC, "dual_rel": DUAL_REL_TOL},
            "verified": not reasons, "reasons": reasons}
